"""Paired comparisons with eb_old_only as the sole reference method."""
import numpy as np
import pandas as pd

KEYS = ["seed", "fold", "benchmark", "subject_key"]
BUDGETS = [0, 1, 3, 7, 15, 31]
BCOLS = [f"B{n}" for n in BUDGETS]
ALC_WEIGHTS = np.array([.1, .2, .2, .2, .2, .1])


def align(tables):
    aligned = {}
    for name, table in tables.items():
        required = KEYS + BCOLS + ["ALC", "n_eval_items", "n_acquired"]
        if table.empty or table[required].isna().any().any() or table.duplicated(KEYS).any():
            raise ValueError(f"missing or duplicate result rows: {name}")
        scores = table[BCOLS + ["ALC"]].to_numpy()
        if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise ValueError(f"invalid scores: {name}")
        if not np.allclose(table["ALC"], table[BCOLS].to_numpy() @ ALC_WEIGHTS, atol=1e-12, rtol=0):
            raise ValueError(f"incorrect ALC weighting: {name}")
        if not table.n_acquired.eq(31).all() or not table.n_eval_items.ge(40).all():
            raise ValueError(f"incomplete acquisition or evaluation: {name}")
        aligned[name] = table.set_index(KEYS).sort_index()
    baseline = aligned["eb_old_only"]
    for name, table in aligned.items():
        if not table.index.equals(baseline.index):
            raise ValueError(f"unmatched evaluation pairs: {name}")
        if not table[["n_eval_items", "n_acquired"]].equals(baseline[["n_eval_items", "n_acquired"]]):
            raise ValueError(f"unmatched acquisition or evaluation counts: {name}")
    return aligned


def compare(tables, run, benchmark_groups):
    aligned = align(tables)
    baseline = aligned["eb_old_only"]
    benches = baseline.index.get_level_values("benchmark").to_numpy()
    unique = np.unique(benches)
    if not set(unique) <= set(benchmark_groups):
        raise ValueError("Missing overlap-group assignment for a scored benchmark")
    clusters = np.array([benchmark_groups[b] for b in benches])
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 2:
        raise ValueError("At least two scored overlap groups are required for the bootstrap")
    positions = [np.flatnonzero(clusters == g) for g in unique_clusters]
    counts = np.array([len(p) for p in positions])
    draws = np.random.default_rng(20261021).integers(0, len(unique_clusters), size=(2000, len(unique_clusters)))
    summary, by_seed, by_fold, by_budget, by_benchmark = [], [], [], [], []
    by_benchmark_budget, by_benchmark_seed, pair_balanced = [], [], []
    for name, table in aligned.items():
        diff = table.ALC - baseline.ALC
        sums = np.array([diff.iloc[p].sum() for p in positions])
        boots = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        summary.append(dict(method=name, ALC=table.ALC.mean(), delta=diff.mean(),
                            ci95_low=lo, ci95_high=hi, win_fraction=(diff < -1e-9).mean()))
        pair_balanced.append(dict(method=name,
                                  ALC=table.ALC.groupby(level=["benchmark", "subject_key"]).mean().mean(),
                                  delta=diff.groupby(level=["benchmark", "subject_key"]).mean().mean()))
        for seed, values in diff.groupby(level="seed"):
            by_seed.append(dict(method=name, seed=seed, ALC=table.xs(seed, level="seed").ALC.mean(),
                                delta=values.mean(), rows=len(values)))
        for fold, values in diff.groupby(level="fold"):
            by_fold.append(dict(method=name, fold=fold, delta=values.mean(), rows=len(values)))
        for bench, values in diff.groupby(level="benchmark"):
            section = table.xs(bench, level="benchmark")
            reference = baseline.xs(bench, level="benchmark")
            by_benchmark.append(dict(method=name, benchmark=bench, ALC=section.ALC.mean(),
                                     delta=values.mean(), rows=len(values),
                                     distinct_pairs=section.index.get_level_values("subject_key").nunique(),
                                     global_delta_contribution=values.sum() / len(diff),
                                     seeds_worse=int((values.groupby(level="seed").mean() > 1e-9).sum())))
            for budget in BCOLS:
                by_benchmark_budget.append(dict(method=name, benchmark=bench, budget=int(budget[1:]),
                                                Brier=section[budget].mean(), rows=len(section),
                                                delta=(section[budget] - reference[budget]).mean()))
            for seed, seed_values in values.groupby(level="seed"):
                by_benchmark_seed.append(dict(method=name, benchmark=bench, seed=seed,
                                              delta=seed_values.mean(), rows=len(seed_values)))
        row = {"method": name}
        for b in BCOLS:
            row[b] = table[b].mean()
            row["delta_" + b] = (table[b] - baseline[b]).mean()
        by_budget.append(row)
    frames = {"summary": pd.DataFrame(summary).sort_values("ALC"),
              "by_seed": pd.DataFrame(by_seed), "by_fold": pd.DataFrame(by_fold),
              "by_budget": pd.DataFrame(by_budget), "by_benchmark": pd.DataFrame(by_benchmark),
              "by_benchmark_budget": pd.DataFrame(by_benchmark_budget),
              "by_benchmark_seed": pd.DataFrame(by_benchmark_seed),
              "pair_balanced_summary": pd.DataFrame(pair_balanced)}
    frames["benchmark_macro_summary"] = pd.DataFrame([
        {"method": name, "ALC": table.ALC.groupby(level="benchmark").mean().mean(),
         "delta": (table.ALC - baseline.ALC).groupby(level="benchmark").mean().mean()}
        for name, table in aligned.items()])
    for name, frame in frames.items():
        frame.to_csv(run / (name + ".csv"), index=False)
    pairs = baseline.reset_index().drop_duplicates(["benchmark", "subject_key"])
    lines = [f"Rows: {len(baseline)} / distinct pairs: {len(pairs)} / benchmarks: {len(unique)} / overlap groups: {len(unique_clusters)}",
             "delta = candidate - eb_old_only; negative is better. One fixed configuration across budgets.",
             frames["summary"].to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             "\nPer-seed results:",
             frames["by_seed"].to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             "\nPer-fold delta (seeds pooled):",
             frames["by_fold"].pivot(index="method", columns="fold", values="delta").to_string(float_format=lambda x: f"{x:+.6f}"),
             "\nPer-budget Brier scores:",
             frames["by_budget"][["method"] + BCOLS].to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             "\nSecondary sensitivity: average seeds within each unique pair first (different weighting):",
             frames["pair_balanced_summary"].to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             "\nSecondary sensitivity: equal weight per benchmark (different estimand):",
             frames["benchmark_macro_summary"].to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             "\nCIs: exploratory overlap-component cluster bootstrap (2,000 resamples).",
             "No adjustment for repeated comparisons or candidate selection; clusters are few.",
             "Rows from repeated seeds are not independent new pairs. win_fraction is a fraction of paired rows.",
             "Snapshot seeds change sampling and labels, not benchmark folds.",
             "Exactly the preceding new-benchmark grouped folds and snapshots are reused; this is a development diagnostic.",
             "The calibration was proposed after inspecting earlier results. These reused public data are not an untouched final test.",
             "Mean matching also changes variance and tail weights; it is not a pure mean-only intervention.",
             "Overlapping training sets across folds are an additional dependence not fully captured by this bootstrap.",
             "No automatic selection, release, submission, or Git modification was performed."]
    for method in ["eb_expanded", "eb_expanded_meanmatch"]:
        changes = frames["by_benchmark"].query("method == @method").sort_values("global_delta_contribution")
        if changes.empty:
            continue
        columns = ["benchmark", "distinct_pairs", "delta", "global_delta_contribution", "seeds_worse"]
        lines.extend([f"\n{method}: five lowest benchmark contributions to overall delta:",
                      changes[columns].head(5).to_string(index=False, float_format=lambda x: f"{x:+.6f}"),
                      f"{method}: five highest benchmark contributions to overall delta:",
                      changes[columns].tail(5).to_string(index=False, float_format=lambda x: f"{x:+.6f}")])
    report = "\n".join(lines) + "\n"
    (run / "comparison.txt").write_text(report, encoding="utf-8")
    print(report, flush=True)


def calibration_contrast(tables, run, benchmark_groups):
    """Secondary, explicitly named contrast against the uncalibrated expansion."""
    aligned = align(tables)
    base = aligned['eb_expanded']
    candidate = aligned['eb_expanded_meanmatch']
    difference = candidate.ALC - base.ALC
    groups = np.array([benchmark_groups[b] for b in base.index.get_level_values('benchmark')])
    unique = np.unique(groups)
    positions = [np.flatnonzero(groups == g) for g in unique]
    counts = np.array([len(p) for p in positions])
    sums = np.array([difference.iloc[p].sum() for p in positions])
    draws = np.random.default_rng(20261021).integers(0, len(unique), size=(2000, len(unique)))
    boots = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    summary = pd.DataFrame([{'candidate': 'eb_expanded_meanmatch', 'reference': 'eb_expanded',
        'delta': difference.mean(), 'ci95_low': low, 'ci95_high': high,
        'win_fraction': (difference < -1e-9).mean()}])
    budget = pd.DataFrame([{'budget': n, 'Brier_delta': (candidate[c] - base[c]).mean(),
                           'ALC_delta_contribution': w * (candidate[c] - base[c]).mean()}
                          for n, c, w in zip(BUDGETS, BCOLS, ALC_WEIGHTS)])
    summary.to_csv(run / 'meanmatch_vs_expanded.csv', index=False)
    budget.to_csv(run / 'meanmatch_vs_expanded_by_budget.csv', index=False)
    text = ('\nSecondary contrast: mean-matched minus uncalibrated expanded prior:\n'
            + summary.to_string(index=False, float_format=lambda x: f'{x:.6f}') + '\n'
            + budget.to_string(index=False, float_format=lambda x: f'{x:+.6f}') + '\n')
    with (run / 'comparison.txt').open('a', encoding='utf-8') as stream:
        stream.write(text)
    print(text, flush=True)
