"""Aggregate test results across seeds: mean, std and confidence interval.

Expected layout:
    <root>/s<SEED>/models/test_result/version_<V>/test_metrics.csv
    <root>/s<SEED>/models/test_result/version_<V>/per_instance_recall.csv

Each version is treated as a model configuration and each seed as an
independent replicate. Statistics are computed across seeds per version.

Usage:
    python -m pugs_detection.aggregate_seeds
    python -m pugs_detection.aggregate_seeds --root . --versions 0 1 --seeds 1 7 10 --ci 0.95
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PATH_RE = re.compile(r"[\\/]s(?P<seed>\d+)[\\/]models[\\/]test_result[\\/]version_(?P<version>\d+)[\\/]")


def collect(root, filename, seeds=None, versions=None):
    """Read every <filename> under root and tag rows with seed/version."""
    frames = []
    root = Path(root).resolve()
    for path in sorted(root.glob(f"s*/models/test_result/version_*/{filename}")):
        m = PATH_RE.search(str(path))
        if m is None:
            continue
        seed, version = int(m["seed"]), int(m["version"])
        if seeds and seed not in seeds:
            continue
        if versions and version not in versions:
            continue
        df = pd.read_csv(path)
        # drop true negatives, true positives, false negatives, and false positives columns if they exist
        df = df.drop(columns=["True Negative", "True Positive", "False Negative", "False Positive"], errors="ignore")
        df.insert(0, "version", version)
        df.insert(0, "seed", seed)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(values, ci):
    """Mean, sample std and t-based CI of the mean across seeds."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mean = x.mean() if n else np.nan
    std = x.std(ddof=1) if n > 1 else np.nan
    if n > 1:
        half = stats.t.ppf((1 + ci) / 2, df=n - 1) * std / np.sqrt(n)
    else:
        half = np.nan
    return pd.Series({
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_half_width": half,
    })


def aggregate(df, group_cols, metric_cols, ci):
    long = df.melt(id_vars=["seed"] + group_cols, value_vars=metric_cols,
                   var_name="metric", value_name="value")
    out = (long.groupby(group_cols + ["metric"], sort=False)["value"]
               .apply(lambda v: summarize(v, ci))
               .unstack()
               .reset_index())
    seeds = (long.groupby(group_cols + ["metric"], sort=False)["seed"]
                 .apply(lambda s: ",".join(map(str, sorted(s.unique()))))
                 .rename("seeds").reset_index())
    return out.merge(seeds, on=group_cols + ["metric"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--seeds", type=int, nargs="*", help="seeds to include (default: all found)")
    p.add_argument("--versions", type=int, nargs="*", default=[0, 1], help="versions to include")
    p.add_argument("--ci", type=float, default=0.95, help="confidence level")
    p.add_argument("--out", type=Path, default=None, help="output dir (default: <root>/aggregated_results)")
    args = p.parse_args()

    out_dir = args.out or args.root / "aggregated_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pct = int(round(args.ci * 100))

    # --- Global test metrics ---
    tm = collect(args.root, "test_metrics.csv", args.seeds, args.versions)
    if tm.empty:
        print("No test_metrics.csv found.")
    else:
        metric_cols = [c for c in tm.columns if c not in ("seed", "version")]
        tm.to_csv(out_dir / "test_metrics_all_runs.csv", index=False)
        summ = aggregate(tm, ["version"], metric_cols, args.ci)
        summ.to_csv(out_dir / "test_metrics_summary.csv", index=False)
        print(f"\n=== Test metrics (mean +/- std, {pct}% CI) ===")
        print_table(summ, ["version", "metric"])

    # --- Per-instance recall per area group ---
    pr = collect(args.root, "per_instance_recall.csv", args.seeds, args.versions)
    if pr.empty:
        print("No per_instance_recall.csv found.")
    else:
        group_order = list(dict.fromkeys(pr["area_group"]))
        pr.to_csv(out_dir / "per_instance_recall_all_runs.csv", index=False)
        summ = aggregate(pr, ["version", "area_group"], ["recall", "n_detected", "n"], args.ci)
        summ["area_group"] = pd.Categorical(summ["area_group"], categories=group_order, ordered=True)
        summ = summ.sort_values(["version", "metric", "area_group"]).reset_index(drop=True)
        summ.to_csv(out_dir / "per_instance_recall_summary.csv", index=False)
        print(f"\n=== Per-instance recall by area group (mean +/- std, {pct}% CI) ===")
        print_table(summ[summ["metric"] == "recall"], ["version", "area_group"])

        # Overall instance recall per run (all groups pooled), then across seeds
        pooled = (pr.groupby(["seed", "version"])[["n", "n_detected"]].sum().reset_index())
        pooled["instance_recall_overall"] = pooled["n_detected"] / pooled["n"]
        summ = aggregate(pooled, ["version"], ["instance_recall_overall"], args.ci)
        summ.to_csv(out_dir / "instance_recall_overall_summary.csv", index=False)
        print(f"\n=== Overall instance recall (pooled over groups) ===")
        print_table(summ, ["version", "metric"])

    print(f"\nSaved results to: {out_dir}")


def print_table(summ, keys):
    view = summ[keys + ["n_seeds"]].astype({"n_seeds": int})
    view["mean +/- std"] = [f"{m:.4f} +/- {s:.4f}" for m, s in zip(summ["mean"], summ["std"])]
    view["CI"] = [f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(summ["ci_low"], summ["ci_high"])]
    view["seeds"] = summ["seeds"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(view.to_string(index=False))


if __name__ == "__main__":
    main()
