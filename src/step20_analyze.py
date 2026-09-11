"""Aggregate SUE Step20 3-seed results and draw curves."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "step20"


def inverse_pairs(x, y, target):
    order = np.argsort(x)
    x = np.asarray(x, float)[order]
    y = np.asarray(y, float)[order]
    valid = np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return np.nan, "insufficient_baseline"

    y = np.maximum.accumulate(y)
    if target < y[0]:
        return np.nan, "below_baseline_range"
    if target > y[-1]:
        return np.nan, "above_baseline_range"

    keep = np.r_[True, np.diff(y) > 1e-12]
    x, y = x[keep], y[keep]
    if len(y) == 1:
        return float(x[0]), "flat_baseline"
    return float(np.interp(target, y, x)), "interpolated"


def aggregate(df):
    keys = ["dataset", "mode", "condition", "n_real", "n_pseudo"]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in ("seed", "n_real", "n_pseudo")]

    summary = (
        df.groupby(keys, as_index=False)[numeric]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [
        "_".join(str(x) for x in c if str(x) != "").rstrip("_")
        if isinstance(c, tuple) else c
        for c in summary.columns
    ]
    return summary


def get_row(summary, condition, n_real, n_pseudo=None):
    q = summary[
        (summary["condition"] == condition)
        & (summary["n_real"] == n_real)
    ]
    if n_pseudo is not None:
        q = q[q["n_pseudo"] == n_pseudo]
    return None if q.empty else q.iloc[0]


def replacement_metrics(summary, n_pseudo):
    rows = []
    base = summary[
        summary["condition"] == "real_only"
    ].sort_values("n_real")

    for stage in ("cca", "final"):
        metric = f"{stage}_mR_mean"
        if metric not in summary.columns:
            continue

        bx = base["n_real"].to_numpy(float)
        by = base[metric].to_numpy(float)

        relation = summary[
            (summary["condition"] == "relation")
            & (summary["n_pseudo"] == n_pseudo)
        ].sort_values("n_real")

        for _, r in relation.iterrows():
            n_real = int(r["n_real"])
            rel = float(r[metric])
            b = get_row(summary, "real_only", n_real)
            o = get_row(summary, "oracle", n_real, n_pseudo)
            s = get_row(summary, "shuffled", n_real, n_pseudo)

            base_perf = np.nan if b is None else float(b[metric])
            oracle = np.nan if o is None else float(o[metric])
            shuffled = np.nan if s is None else float(s[metric])

            eq, status = (
                (np.nan, "relation_unavailable")
                if not np.isfinite(rel)
                else inverse_pairs(bx, by, rel)
            )

            gain = rel - base_perf
            prr = (
                (eq - n_real) / n_pseudo
                if np.isfinite(eq) else np.nan
            )
            reduction = (
                1.0 - n_real / eq
                if np.isfinite(eq) and eq > 0 else np.nan
            )
            oracle_gap = oracle - base_perf
            ogr = (
                gain / oracle_gap
                if np.isfinite(oracle_gap)
                and abs(oracle_gap) > 1e-12
                else np.nan
            )

            rows.append({
                "stage": stage,
                "n_real": n_real,
                "n_pseudo": n_pseudo,
                "real_mR": base_perf,
                "relation_mR": rel,
                "shuffled_mR": shuffled,
                "oracle_mR": oracle,
                "relation_gain": gain,
                "relation_vs_shuffled": rel - shuffled,
                "pair_equivalent_number": eq,
                "pair_equivalent_status": status,
                "pair_replacement_ratio": prr,
                "pair_reduction_rate": reduction,
                "oracle_gap_recovery": ogr,
            })
    return pd.DataFrame(rows)


def plot_curves(summary, out_dir, n_pseudo):
    labels = {
        "real_only": "Real-only",
        "relation": "Relation",
        "shuffled": "Shuffled",
        "oracle": "Oracle",
    }

    for stage in ("cca", "final"):
        for metric in ("mean_R1", "mean_R5", "mean_R10", "mR"):
            mean_col = f"{stage}_{metric}_mean"
            std_col = f"{stage}_{metric}_std"
            if mean_col not in summary.columns:
                continue

            fig, ax = plt.subplots(figsize=(7, 5))
            plotted = False

            for condition in labels:
                q = summary[summary["condition"] == condition].copy()
                if condition != "real_only":
                    q = q[q["n_pseudo"] == n_pseudo]
                q = q.sort_values("n_real")
                q = q[np.isfinite(q[mean_col])]
                if q.empty:
                    continue

                yerr = (
                    q[std_col].fillna(0).to_numpy()
                    if std_col in q.columns else None
                )
                ax.errorbar(
                    q["n_real"],
                    q[mean_col],
                    yerr=yerr,
                    marker="o",
                    capsize=3,
                    label=labels[condition],
                )
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_xlabel("Number of real pairs")
            ax.set_ylabel(metric)
            ax.set_title(f"{stage.upper()} | {metric} | pseudo={n_pseudo}")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                out_dir / f"{stage}_{metric}_pseudo{n_pseudo}.png",
                dpi=180,
            )
            plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", help="flickr30 or mscoco")
    p.add_argument("--mode", choices=("main", "saturation"), default="main")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_pseudo", type=int, default=75)
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()

    base = Path(args.output_dir) / args.mode / args.data
    frames = []
    for seed in args.seeds:
        path = base / f"seed{seed}" / "results.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))

    df = pd.concat(frames, ignore_index=True)
    summary = aggregate(df)

    analysis_dir = base / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(analysis_dir / "summary.csv", index=False)

    metrics = replacement_metrics(summary, args.n_pseudo)
    metrics.to_csv(
        analysis_dir / f"replacement_metrics_pseudo{args.n_pseudo}.csv",
        index=False,
    )

    plot_curves(summary, analysis_dir, args.n_pseudo)

    print(f"Saved: {analysis_dir / 'summary.csv'}")
    print(
        "Saved:",
        analysis_dir / f"replacement_metrics_pseudo{args.n_pseudo}.csv",
    )
    print(f"Plots: {analysis_dir}")


if __name__ == "__main__":
    main()
