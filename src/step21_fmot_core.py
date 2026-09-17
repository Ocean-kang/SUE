import math
import torch
import torch.nn.functional as F


def make_strict_unpaired(train_set, seed=0, removal=0.1):
    """Independent sampling/shuffling. No x[i] <-> y[i] relation is used."""
    x, y = train_set
    n = min(len(x), len(y))
    keep = int(n * (1.0 - removal))

    gx = torch.Generator().manual_seed(seed * 2 + 1)
    gy = torch.Generator().manual_seed(seed * 2 + 2)
    ix = torch.randperm(n, generator=gx)[:keep]
    iy = torch.randperm(n, generator=gy)[:keep]
    return (x[ix], y[iy])


def _as_tensor(z, device):
    if isinstance(z, torch.Tensor):
        return z.to(device=device, dtype=torch.float32)
    return torch.as_tensor(z, device=device, dtype=torch.float32)


def spectral_transform(net, x):
    net.transform(x.float())
    return torch.from_numpy(net.embeddings_).float()


def prepare_basis(net, original_x, z, support_idx, device):
    """
    Use SUE's own graph construction to estimate eigenvalues by Rayleigh quotient.
    Returns the normalized/sorted full basis plus support basis and preprocessing state.
    """
    z = _as_tensor(z, device)
    scale = z.square().mean(0).sqrt().clamp_min(1e-8)
    phi = z / scale

    idx = support_idx.to(device)
    phi_sub = phi[idx]

    x_sub = original_x[support_idx.cpu()].float().to(net.device)
    L = torch.as_tensor(net._build_laplacian(x_sub), device=device, dtype=phi.dtype)

    Lphi = L @ phi_sub
    evals = (phi_sub * Lphi).sum(0) / phi_sub.square().sum(0).clamp_min(1e-8)
    evals = evals.clamp_min(0)

    perm = torch.argsort(evals)
    evals = evals[perm]
    phi = phi[:, perm]
    phi_sub = phi_sub[:, perm]
    scale = scale[perm]

    gram = phi_sub.T @ phi_sub / phi_sub.shape[0]
    orth_error = torch.linalg.norm(
        gram - torch.eye(gram.shape[0], device=device), ord="fro"
    ).item()

    return {
        "phi": phi,
        "phi_sub": phi_sub,
        "evals": evals,
        "perm": perm,
        "scale": scale,
        "orth_error": orth_error,
    }


def preprocess_test(z, state, device):
    z = _as_tensor(z, device)
    return (z / state["scale"])[..., state["perm"]]


def joint_hks_times(evals_x, evals_y, n_hks=32):
    vals = torch.cat([evals_x, evals_y])
    vals = vals[vals > 1e-6]
    if vals.numel() == 0:
        return torch.logspace(-2, 2, n_hks, device=evals_x.device)

    lmin = vals.min()
    lmax = vals.max().clamp_min(lmin + 1e-6)
    t_min = 4.0 * math.log(10.0) / lmax
    t_max = 4.0 * math.log(10.0) / lmin
    t_max = torch.minimum(t_max, t_min * 1e4)

    return torch.exp(
        torch.linspace(torch.log(t_min), torch.log(t_max), n_hks, device=evals_x.device)
    )


def hks(phi, evals, times):
    weights = torch.exp(-evals[:, None] * times[None, :])
    desc = phi.square() @ weights
    desc = torch.log(desc.clamp_min(1e-8))
    return (desc - desc.mean(0, keepdim=True)) / desc.std(0, keepdim=True).clamp_min(1e-6)


def descriptor_coefficients(phi, desc):
    return phi.T @ desc / phi.shape[0]


def solve_fm(A, B, evals_x, evals_y, lap_weight=1.0, reg=1e-4):
    """Solve min_C ||CA-B||^2 + lap_weight * ||Ly C - C Lx||^2."""
    k = A.shape[0]
    eye = torch.eye(k, device=A.device, dtype=A.dtype)
    AAt = A @ A.T
    rows = []

    for i in range(k):
        freq = (evals_y[i] - evals_x).square()
        lhs = AAt + lap_weight * torch.diag(freq) + reg * eye
        rhs = A @ B[i]
        rows.append(torch.linalg.solve(lhs, rhs))

    return torch.stack(rows, 0)


def soft_correspondence(phi_x, phi_y, Cxy, temperature=0.07):
    x_to_y = F.normalize(phi_x @ Cxy.T, dim=1)
    y = F.normalize(phi_y, dim=1)
    return torch.softmax((x_to_y @ y.T) / temperature, dim=1)


def sliced_wasserstein(x, y, n_proj=64, p=2):
    """Compact equal-mass SW. x/y must have the same number of support samples."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("SW support sizes must match.")
    theta = torch.randn(x.shape[1], n_proj, device=x.device, dtype=x.dtype)
    theta = F.normalize(theta, dim=0)
    xp = torch.sort(x @ theta, dim=0).values
    yp = torch.sort(y @ theta, dim=0).values
    return (xp.sub(yp).abs().pow(p).mean() + 1e-12).pow(1.0 / p)


def fm_losses(Cxy, Cyx, A, B, lx, ly):
    k = Cxy.shape[0]
    eye = torch.eye(k, device=Cxy.device, dtype=Cxy.dtype)

    desc = (Cxy @ A - B).square().mean() + (Cyx @ B - A).square().mean()
    lap = (ly[:, None] * Cxy - Cxy * lx[None, :]).square().mean()
    lap = lap + (lx[:, None] * Cyx - Cyx * ly[None, :]).square().mean()

    bij = (Cxy @ Cyx - eye).square().mean() + (Cyx @ Cxy - eye).square().mean()
    orth = (Cxy.T @ Cxy - eye).square().mean() + (Cyx.T @ Cyx - eye).square().mean()
    return desc, lap, bij, orth


def proper_targets(phi_x, phi_y, Pxy, Pyx):
    # Convention: phi_x @ Cxy.T ~= Pxy @ phi_y
    pinv_x = torch.linalg.pinv(phi_x)
    pinv_y = torch.linalg.pinv(phi_y)
    Cxy = (pinv_x @ (Pxy.detach() @ phi_y)).T
    Cyx = (pinv_y @ (Pyx.detach() @ phi_x)).T
    return Cxy, Cyx


def refine_fm(Cxy0, Cyx0, phi_x, phi_y, A, B, lx, ly, cfg):
    Cxy = torch.nn.Parameter(Cxy0.clone())
    Cyx = torch.nn.Parameter(Cyx0.clone())
    opt = torch.optim.Adam([Cxy, Cyx], lr=cfg["lr"])

    for step in range(1, cfg["steps"] + 1):
        Pxy = soft_correspondence(phi_x, phi_y, Cxy, cfg["temperature"])
        Pyx = soft_correspondence(phi_y, phi_x, Cyx, cfg["temperature"])

        desc, lap, bij, orth = fm_losses(Cxy, Cyx, A, B, lx, ly)

        target_xy, target_yx = proper_targets(phi_x, phi_y, Pxy, Pyx)
        proper = (Cxy - target_xy).square().mean() + (Cyx - target_yx).square().mean()

        x_to_y = phi_x @ Cxy.T
        y_to_x = phi_y @ Cyx.T
        ot = sliced_wasserstein(
            x_to_y, Pxy @ phi_y, cfg["ot_projections"], cfg["ot_p"]
        )
        ot = ot + sliced_wasserstein(
            y_to_x, Pyx @ phi_x, cfg["ot_projections"], cfg["ot_p"]
        )

        loss = (
            cfg["w_desc"] * desc
            + cfg["w_lap"] * lap
            + cfg["w_bij"] * bij
            + cfg["w_orth"] * orth
            + cfg["w_proper"] * proper
            + cfg["w_ot"] * ot
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([Cxy, Cyx], cfg["grad_clip"])
        opt.step()

        if step == 1 or step % cfg["log_every"] == 0 or step == cfg["steps"]:
            with torch.no_grad():
                entropy = -(Pxy * Pxy.clamp_min(1e-8).log()).sum(1).mean()
                cycle = torch.linalg.norm(Cxy @ Cyx - torch.eye(Cxy.shape[0], device=Cxy.device))
                sv = torch.linalg.svdvals(Cxy)
            print(
                f"[{step:04d}/{cfg['steps']}] "
                f"loss={loss.item():.5f} desc={desc.item():.5f} "
                f"lap={lap.item():.5f} proper={proper.item():.5f} "
                f"ot={ot.item():.5f} H(P)={entropy.item():.3f} "
                f"cycle={cycle.item():.3f} sv=[{sv.min().item():.3f},{sv.max().item():.3f}]"
            )

    return Cxy.detach(), Cyx.detach()


@torch.no_grad()
def evaluate_retrieval(zx, zy, Cxy=None, Cyx=None):
    """
    modality1=image, modality2=text (same convention as SUE retrieval.py).
    """
    if Cxy is None:
        x_query = F.normalize(zx, dim=1)
        y_query = F.normalize(zy, dim=1)
        sim_i2t = x_query @ y_query.T
        sim_t2i = y_query @ x_query.T
    else:
        x_to_y = F.normalize(zx @ Cxy.T, dim=1)
        y_to_x = F.normalize(zy @ Cyx.T, dim=1)
        sim_i2t = x_to_y @ F.normalize(zy, dim=1).T
        sim_t2i = y_to_x @ F.normalize(zx, dim=1).T

    from general_utils import calc_recall

    i2t = calc_recall(sim_i2t.cpu())
    t2i = calc_recall(sim_t2i.cpu())
    out = {
        "i2t": [float(v) for v in i2t],
        "t2i": [float(v) for v in t2i],
    }
    out["mR"] = sum(out["i2t"] + out["t2i"]) / 6.0
    return out
