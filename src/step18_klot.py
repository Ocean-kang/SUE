"""SUE Step18 - KLOT pair-information amplification.

Fixed SE -> CCA teacher -> pair-only control / pair+KLOT -> optional unchanged SUE MMD.
Modes: pair sweep, pair corruption, pair coverage.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pair_removal import (
    compute_bidirectional_recall,
    load_config,
    load_fixed_se_cache,
    print_recall,
    run_mmd,
    save_results_csv,
    set_seed,
)
from step17_functional_map import fit_cross_cca, project, recall_fields

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "step18"
DEFAULT_PAIRS = [500, 300, 200, 175, 150, 125, 100, 75, 50, 25, 10]


def cosine_matrix(x, y):
    return F.normalize(x, dim=1) @ F.normalize(y, dim=1).T


def sinkhorn_plan(similarity, epsilon, steps):
    """Uniform-marginal entropic OT in log space."""
    n, m = similarity.shape
    log_k = similarity / epsilon
    log_a, log_b = -np.log(n), -np.log(m)
    u = torch.zeros(n, dtype=similarity.dtype, device=similarity.device)
    v = torch.zeros(m, dtype=similarity.dtype, device=similarity.device)
    for _ in range(steps):
        u = log_a - torch.logsumexp(log_k + v[None], dim=1)
        v = log_b - torch.logsumexp(log_k + u[:, None], dim=0)
    # ponytail: one symmetric correction matches the public KLOT solver closely.
    u2 = log_a - torch.logsumexp(log_k + v[None], dim=1)
    v2 = log_b - torch.logsumexp(log_k + u[:, None], dim=0)
    u, v = 0.5 * (u + u2), 0.5 * (v + v2)
    log_p = log_k + u[:, None] + v[None]
    return torch.exp(log_p), log_p


class KLOTLoss(nn.Module):
    """KLOT value with SOTAlign's explicit student-affinity gradient."""

    def __init__(self, eps_student=0.05, eps_teacher=0.005, steps=100):
        super().__init__()
        self.eps_student = eps_student
        self.eps_teacher = eps_teacher
        self.steps = steps

    def forward(self, student_sim, teacher_sim):
        with torch.no_grad():
            ps, log_ps = sinkhorn_plan(student_sim, self.eps_student, self.steps)
            pt, log_pt = sinkhorn_plan(teacher_sim, self.eps_teacher, self.steps)
            value = (pt * (log_pt - log_ps)).sum()
        proxy = ((ps - pt).detach() * student_sim).sum() / self.eps_student
        return value.detach() + proxy - proxy.detach()


def infonce(z1, z2, temperature):
    logits = cosine_matrix(z1, z2) / temperature
    labels = torch.arange(len(z1), device=z1.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


def optimize_student(
    train1,
    train2,
    pair_idx1,
    pair_idx2,
    unpaired_idx,
    w1_init,
    w2_init,
    args,
    device,
    seed,
    klot_weight,
):
    """CCA-initialized linear student; KLOT is the only difference vs control."""
    set_seed(seed)
    rng = np.random.default_rng(seed + 18000)
    x = torch.as_tensor(train1, dtype=torch.float32, device=device)
    y = torch.as_tensor(train2, dtype=torch.float32, device=device)
    w1_0 = torch.as_tensor(w1_init, dtype=torch.float32, device=device)
    w2_0 = torch.as_tensor(w2_init, dtype=torch.float32, device=device)
    w1 = torch.nn.Parameter(w1_0.clone())
    w2 = torch.nn.Parameter(w2_0.clone())
    optimizer = torch.optim.Adam([w1, w2], lr=args.lr)
    klot = KLOTLoss(args.epsilon_student, args.epsilon_teacher, args.sinkhorn_steps)

    pair_idx1 = np.asarray(pair_idx1, dtype=np.int64)
    pair_idx2 = np.asarray(pair_idx2, dtype=np.int64)
    unpaired_idx = np.asarray(unpaired_idx, dtype=np.int64)
    tail = []

    for _ in range(args.student_steps):
        k = min(args.pair_batch_size, len(pair_idx1))
        pos = rng.choice(len(pair_idx1), size=k, replace=False)
        pi = torch.as_tensor(pair_idx1[pos], device=device)
        pt = torch.as_tensor(pair_idx2[pos], device=device)

        b = min(args.unpaired_batch_size, len(unpaired_idx))
        ui = torch.as_tensor(rng.choice(unpaired_idx, b, replace=False), device=device)
        ut = torch.as_tensor(rng.choice(unpaired_idx, b, replace=False), device=device)

        pair_loss = infonce(x[pi] @ w1, y[pt] @ w2, args.temperature)
        klot_loss = torch.zeros((), device=device)
        if klot_weight > 0:
            with torch.no_grad():
                teacher_sim = cosine_matrix(x[ui] @ w1_0, y[ut] @ w2_0)
            student_sim = cosine_matrix(x[ui] @ w1, y[ut] @ w2)
            klot_loss = klot(student_sim, teacher_sim)

        loss = pair_loss + klot_weight * klot_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # ponytail: cosine losses ignore scale; preserve CCA column scales for fair MMD.
        with torch.no_grad():
            w1.mul_(w1_0.norm(dim=0, keepdim=True) / w1.norm(dim=0, keepdim=True).clamp_min(1e-12))
            w2.mul_(w2_0.norm(dim=0, keepdim=True) / w2.norm(dim=0, keepdim=True).clamp_min(1e-12))

        tail.append((float(pair_loss.detach()), float(klot_loss.detach())))
        tail = tail[-20:]

    stats = {
        "pair_loss": float(np.mean([v[0] for v in tail])),
        "klot_loss": float(np.mean([v[1] for v in tail])),
    }
    return w1.detach().cpu().numpy(), w2.detach().cpu().numpy(), stats


def transport_diag(z1, z2, epsilon, args, device):
    n = min(args.diag_size, len(z1), len(z2))
    x = torch.as_tensor(z1[:n], dtype=torch.float32, device=device)
    y = torch.as_tensor(z2[:n], dtype=torch.float32, device=device)
    with torch.no_grad():
        p, log_p = sinkhorn_plan(cosine_matrix(x, y), epsilon, args.sinkhorn_steps)
    return float(torch.diagonal(p).sum().item()), float((-(p * log_p).sum() / np.log(n * n)).item())


def teacher_student_kl(teacher, student, args, device):
    n = min(args.diag_size, len(teacher["test1"]))
    t1 = torch.as_tensor(teacher["test1"][:n], dtype=torch.float32, device=device)
    t2 = torch.as_tensor(teacher["test2"][:n], dtype=torch.float32, device=device)
    s1 = torch.as_tensor(student["test1"][:n], dtype=torch.float32, device=device)
    s2 = torch.as_tensor(student["test2"][:n], dtype=torch.float32, device=device)
    with torch.no_grad():
        pt, log_pt = sinkhorn_plan(cosine_matrix(t1, t2), args.epsilon_teacher, args.sinkhorn_steps)
        _, log_ps = sinkhorn_plan(cosine_matrix(s1, s2), args.epsilon_student, args.sinkhorn_steps)
    return float((pt * (log_pt - log_ps)).sum().item())


def corrupt_text_indices(indices, rate, rng):
    out = indices.copy()
    if rate == 0:
        return out
    k = min(len(out), max(2, int(round(len(out) * rate))))
    pos = rng.choice(len(out), k, replace=False)
    out[pos] = np.roll(out[pos], int(rng.integers(1, k)))
    return out


def farthest_pairs(features, candidates, n, rng):
    # ponytail: exact O(500^2) FPS is tiny here; use ANN/k-center only for larger pools.
    x = features[candidates].astype(np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    first = int(rng.integers(len(candidates)))
    selected = [first]
    best = 1.0 - x @ x[first]
    best[first] = -np.inf
    for _ in range(1, n):
        nxt = int(np.argmax(best))
        selected.append(nxt)
        best = np.minimum(best, 1.0 - x @ x[nxt])
        best[np.asarray(selected)] = -np.inf
    return candidates[np.asarray(selected)]


def coverage_pairs(features, pair_order, n, mode, rng):
    if mode == "random":
        return rng.choice(pair_order, n, replace=False)
    if mode == "high":
        return farthest_pairs(features, pair_order, n, rng)
    x = features[pair_order].astype(np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    center = int(rng.integers(len(pair_order)))
    return pair_order[np.argsort(-(x @ x[center]))[:n]]


def coverage_score(features, pair_order, selected):
    pool = features[pair_order].astype(np.float64)
    anchor = features[selected].astype(np.float64)
    pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
    anchor /= np.maximum(np.linalg.norm(anchor, axis=1, keepdims=True), 1e-12)
    return float(np.max(pool @ anchor.T, axis=1).mean())


def conditions(args, train1, pair_order):
    rng = np.random.default_rng(args.seed + 18100)
    if args.mode == "sweep":
        return [(n, pair_order[:n], pair_order[:n], 0.0, "prefix") for n in args.pairs]
    if args.mode == "corruption":
        return [
            (n, pair_order[:n], corrupt_text_indices(pair_order[:n], r, rng), r, "prefix")
            for n in args.pairs for r in args.corruption_rates
        ]
    return [
        (n, idx, idx, 0.0, mode)
        for n in args.pairs for mode in args.coverages
        for idx in [coverage_pairs(train1, pair_order, n, mode, rng)]
    ]


def evaluate_mmd(z, args, device):
    m = run_mmd(
        z["train1"], z["train2"], z["test1"], z["test2"],
        device=device, seed=args.seed, epochs=args.mmd_epochs,
        batch_size=args.mmd_batch_size, n_scales=args.mmd_scales,
    )
    return compute_bidirectional_recall(m["test1"], m["test2"])


def run_condition(args, train1, train2, test1, test2, idx1, idx2,
                  unpaired_idx, n_components, device, corruption, coverage):
    n = len(idx1)
    w1, w2 = fit_cross_cca(train1, train2, idx1, idx2, n_components)
    teacher = project(train1, train2, test1, test2, w1, w2)
    cca = compute_bidirectional_recall(teacher["test1"], teacher["test2"])
    print_recall(f"[{n} pairs] SE -> CCA", cca)

    pw1, pw2, pair_stats = optimize_student(
        train1, train2, idx1, idx2, unpaired_idx, w1, w2,
        args, device, args.seed + n, 0.0,
    )
    pair_only_z = project(train1, train2, test1, test2, pw1, pw2)
    pair_only = compute_bidirectional_recall(pair_only_z["test1"], pair_only_z["test2"])
    print_recall(f"[{n} pairs] CCA-init -> pair-only student", pair_only)

    kw1, kw2, klot_stats = optimize_student(
        train1, train2, idx1, idx2, unpaired_idx, w1, w2,
        args, device, args.seed + n, args.klot_weight,
    )
    klot_z = project(train1, train2, test1, test2, kw1, kw2)
    klot_recall = compute_bidirectional_recall(klot_z["test1"], klot_z["test2"])
    print_recall(f"[{n} pairs] CCA-init -> pair + KLOT", klot_recall)

    cca_mmd = klot_mmd = None
    if not args.skip_mmd:
        cca_mmd = evaluate_mmd(teacher, args, device)
        print_recall(f"[{n} pairs] SE -> CCA -> MMD", cca_mmd)
        klot_mmd = evaluate_mmd(klot_z, args, device)
        print_recall(f"[{n} pairs] SE -> CCA -> KLOT -> MMD", klot_mmd)

    teacher_mass, teacher_entropy = transport_diag(
        teacher["test1"], teacher["test2"], args.epsilon_teacher, args, device
    )
    klot_mass, klot_entropy = transport_diag(
        klot_z["test1"], klot_z["test2"], args.epsilon_student, args, device
    )
    gain = recall_fields("klot", klot_recall)["klot_mean_R10"] - recall_fields(
        "pair_only", pair_only
    )["pair_only_mean_R10"]

    row = {
        "dataset": args.data,
        "seed": args.seed,
        "mode": args.mode,
        "n_real": n,
        "corruption_rate": corruption,
        "pair_correct_rate": 1.0 - corruption,
        "coverage": coverage,
        "student_steps": args.student_steps,
        "klot_weight": args.klot_weight,
        "unpaired_pool": len(unpaired_idx),
        "teacher_ot_diag_mass": teacher_mass,
        "teacher_ot_entropy": teacher_entropy,
        "klot_ot_diag_mass": klot_mass,
        "klot_ot_entropy": klot_entropy,
        "teacher_student_plan_kl": teacher_student_kl(teacher, klot_z, args, device),
        "pair_only_train_loss": pair_stats["pair_loss"],
        "klot_pair_train_loss": klot_stats["pair_loss"],
        "klot_train_loss": klot_stats["klot_loss"],
        "propagation_gain_mean_R10": gain,
        **recall_fields("cca", cca),
        **recall_fields("cca_mmd", cca_mmd),
        **recall_fields("pair_only", pair_only),
        **recall_fields("klot", klot_recall),
        **recall_fields("klot_mmd", klot_mmd),
    }
    return row, (w1, w2, pw1, pw2, kw1, kw2)


def self_check():
    set_seed(18)
    student = torch.randn(8, 8, requires_grad=True)
    teacher = torch.eye(8)
    loss = KLOTLoss(0.1, 0.05, 20)(student, teacher)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    with torch.no_grad():
        p, _ = sinkhorn_plan(torch.eye(8), 0.05, 30)
        target = torch.full((8,), 1 / 8)
        assert torch.allclose(p.sum(0), target, atol=1e-3)
        assert torch.allclose(p.sum(1), target, atol=1e-3)
    print("Step18 self-check passed.")


def parse_args():
    p = argparse.ArgumentParser(description="SUE Step18 - KLOT pair amplification")
    p.add_argument("data", nargs="?", help="e.g. flickr30")
    p.add_argument("--mode", choices=["sweep", "corruption", "coverage"], default="sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, nargs="+", default=None)
    p.add_argument("--corruption_rates", type=float, nargs="+", default=[0, .1, .25, .5, .75, 1.0])
    p.add_argument("--coverages", nargs="+", choices=["random", "high", "low"], default=["random", "high", "low"])
    p.add_argument("--student_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pair_batch_size", type=int, default=128)
    p.add_argument("--unpaired_batch_size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--klot_weight", type=float, default=1.0)
    p.add_argument("--epsilon_student", type=float, default=0.05)
    p.add_argument("--epsilon_teacher", type=float, default=0.005)
    p.add_argument("--sinkhorn_steps", type=int, default=100)
    p.add_argument("--diag_size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mmd_epochs", type=int, default=100)
    p.add_argument("--mmd_batch_size", type=int, default=32)
    p.add_argument("--mmd_scales", type=int, default=3)
    p.add_argument("--skip_mmd", action="store_true")
    p.add_argument("--cache", default=None)
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--self_check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        return args
    if not args.data:
        p.error("data is required unless --self_check is used")
    if args.pairs is None:
        args.pairs = DEFAULT_PAIRS if args.mode == "sweep" else [300, 150, 100]
    if any(not 0 <= r <= 1 for r in args.corruption_rates):
        p.error("corruption_rates must be in [0, 1]")
    return args


def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return

    set_seed(args.seed)
    config = load_config(args.data)
    n_components = int(config["n_components"])
    cache = load_fixed_se_cache(args.data, args.seed, args.cache)
    train1, train2 = cache["train_se1"], cache["train_se2"]
    test1, test2 = cache["test_se1"], cache["test_se2"]
    pair_order = cache["pair_order"]
    if any(n <= n_components or n > len(pair_order) for n in args.pairs):
        raise ValueError(f"pairs must be in [{n_components + 1}, {len(pair_order)}]")

    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    # Exclude all master pairs, including unused ones: KLOT sees unpaired data only.
    unpaired_idx = np.setdiff1d(np.arange(len(train1)), pair_order)
    run_dir = Path(args.output_dir) / args.mode / args.data / f"seed{args.seed}"
    diag_dir = run_dir / "diagnostics"
    run_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(exist_ok=True)

    print(f"Step18 | data={args.data} | mode={args.mode} | seed={args.seed} | device={device}")
    print(f"pairs={args.pairs} | unpaired_pool={len(unpaired_idx)} | skip_mmd={args.skip_mmd}")

    rows = []
    for n, idx1, idx2, corruption, coverage in conditions(args, train1, pair_order):
        suffix = f"corr{int(100 * corruption)}" if args.mode == "corruption" else coverage
        tag = f"N{n}_{suffix}"
        print("\n" + "=" * 72 + f"\n{tag}\n" + "=" * 72)
        row, weights = run_condition(
            args, train1, train2, test1, test2, idx1, idx2,
            unpaired_idx, n_components, device, corruption, coverage,
        )
        row["condition"] = tag
        row["pair_coverage_score"] = coverage_score(train1, pair_order, idx1)
        rows.append(row)
        save_results_csv(rows, run_dir / "results.csv")
        np.savez_compressed(
            diag_dir / f"{tag}.npz",
            pair_image_indices=idx1,
            pair_text_indices=idx2,
            teacher_w1=weights[0], teacher_w2=weights[1],
            pair_only_w1=weights[2], pair_only_w2=weights[3],
            klot_w1=weights[4], klot_w2=weights[5],
        )

    print(f"\nStep18 completed. Results: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
