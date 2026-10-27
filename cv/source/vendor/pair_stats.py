"""Aggregate one benchmark at a time; fit only on the training folds."""
import numpy as np
import pandas as pd


def aggregate_benchmark(responses, benchmark):
    required = ["benchmark_id", "subject_key", "item_id", "response"]
    if responses.empty or responses[required].isna().any().any():
        raise ValueError(f"empty responses or missing fields: {benchmark}")
    if set(responses["benchmark_id"].astype(str)) != {benchmark}:
        raise ValueError("parquet filter returned another benchmark")
    if not responses["response"].isin([0, 1]).all():
        raise ValueError("response must be binary")
    # Exactly the original estimator: trials -> item mean -> pair mean.
    per_item = responses.groupby(["benchmark_id", "subject_key", "item_id"])["response"].mean()
    table = per_item.groupby(["benchmark_id", "subject_key"]).agg(["mean", "size"]).reset_index()
    return table.rename(columns={"size": "n_items"})


def fit_prior(stats, folds, held_out_fold, max_items=None, grid_size=101, smooth=1e-3):
    if held_out_fold != "all" and held_out_fold not in set(folds.values()):
        raise ValueError("unknown held-out fold")
    train = sorted(b for b, f in folds.items() if held_out_fold == "all" or f != held_out_fold)
    table = stats[stats["benchmark_id"].isin(train) & (stats["n_items"] >= 80)]
    if max_items is not None:
        table = table[table["n_items"] <= max_items]
    if table.empty:
        raise ValueError(f"no training pairs in fold {held_out_fold}, cap={max_items}")
    if not np.isfinite(table["mean"]).all() or not table["mean"].between(0, 1).all():
        raise ValueError("invalid training rates")
    grid = np.linspace(0.0, 1.0, grid_size)
    weights = np.zeros(grid_size)
    counts = {}
    for bid, group in table.groupby("benchmark_id"):
        idx = np.rint(group["mean"].to_numpy() * (grid_size - 1)).astype(int)
        hist = np.bincount(idx, minlength=grid_size).astype(float)
        weights += hist / hist.sum()
        counts[str(bid)] = len(group)
    weights += smooth
    weights /= weights.sum()
    return {"grid": grid.round(6).tolist(), "weights": weights.tolist(),
            "fold": str(held_out_fold), "n_benchmarks": len(counts), "n_pairs": len(table),
            "max_items": max_items, "training_benchmarks": sorted(counts),
            "training_pair_counts": counts, "mean": float(grid @ weights),
            "tail0": float(weights[grid <= .02].sum()),
            "tail1": float(weights[grid >= .98].sum())}
