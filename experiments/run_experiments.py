"""CPU-only, paired EB ablations. Never rewrites work/folds.json or submissions/."""
import os

# Set before importing numeric libraries. Limit local CPU parallelism as well.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import local_eval_v2 as evaluator
from pair_stats import aggregate_benchmark, fit_prior

BUDGET_COLUMNS = [f"B{b}" for b in evaluator.BUDGETS]
KEYS = ["seed", "fold", "benchmark", "subject_key"]


def configurations(cap):
    configs = {}
    for name, max_items, mix in [("eb_grid", None, 0.0), ("eb_cap", cap, 0.0),
                                  ("eb_mix25", None, .25), ("eb_cap_mix25", cap, .25)]:
        configs[name] = {"max_items": max_items, "uniform_mix": mix, "zero_half": False}
        zero_name = "eb_zero_half" if name == "eb_grid" else name + "_zero_half"
        configs[zero_name] = {**configs[name], "zero_half": True}
    configs["beta11"] = {"max_items": None, "uniform_mix": 1.0, "zero_half": True}
    return configs


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def data_identity(work):
    out = {}
    for name in ["responses.parquet", "items.parquet", "subjects.parquet"]:
        st = (work / name).stat()
        out[name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    out["folds.json"] = {"sha256": sha256(work / "folds.json")}
    return out


def validate_settings(reference, fold_data, seeds, n_draws=None):
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    old = reference["args"]
    n_folds = int(fold_data["n_folds"])
    if int(old["n_folds"]) != n_folds or int(old["seed"]) != int(fold_data["seed"]):
        raise ValueError("reference JSON and folds.json differ; stopping without overwriting folds")
    folds = fold_data["folds"]
    if n_folds < 2 or set(folds.values()) != set(range(n_folds)):
        raise ValueError("invalid fold inventory")
    settings = {"n_folds": n_folds, "fold_seed": int(fold_data["seed"]),
                "n_draws": int(old["n_draws"]) if n_draws is None else n_draws,
                "draw_limit": int(old["draw_limit"]), "snapshot_seeds": list(seeds)}
    if settings["n_draws"] < 1 or settings["draw_limit"] < 80:
        raise ValueError("invalid snapshot settings")
    return settings


def collect_stats(work, folds, run):
    tables = []
    bids = sorted(folds)
    for i, bid in enumerate(bids, 1):
        responses = pd.read_parquet(work / "responses.parquet",
                                    columns=["benchmark_id", "subject_key", "item_id", "response"],
                                    filters=[("benchmark_id", "in", [bid])])
        table = aggregate_benchmark(responses, bid)
        tables.append(table)
        print(f"集計 {i}/{len(bids)}: {bid}, {len(table)} pairs", flush=True)
        del responses
    stats = pd.concat(tables, ignore_index=True)
    if stats.duplicated(["benchmark_id", "subject_key"]).any():
        raise ValueError("duplicate pair statistics")
    stats.to_csv(run / "pair_stats.csv", index=False)
    return stats


def verify_original_training(code, work, prior, run):
    target = run / "reference_fold0_prior.json"
    command = [sys.executable, "-u", str(code / "reference_train_eb_prior.py"),
               "--work", str(work), "--fold", "0", "--out", str(target)]
    print("現行の学習コードでfold 0を再学習し、集計版との一致を確認します。", flush=True)
    with (run / "reference_train.log").open("w") as f:
        subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT)
    reference = json.loads(target.read_text())
    for key in ["grid", "weights"]:
        if len(prior[key]) != len(reference[key]) or not np.allclose(prior[key], reference[key], atol=1e-12, rtol=0):
            raise ValueError(f"original training differs for {key}; see {target}")
    for key in ["n_pairs", "n_benchmarks"]:
        if prior[key] != reference[key]:
            raise ValueError(f"original training differs for {key}")
    print("現行学習との一致: PASS（grid、weights、学習ペア数、ベンチマーク数）", flush=True)


def align_results(tables):
    aligned = {}
    for name, table in tables.items():
        required = KEYS + BUDGET_COLUMNS + ["ALC", "n_eval_items", "n_acquired"]
        if table.empty or table[required].isna().any().any() or table.duplicated(KEYS).any():
            raise ValueError(f"invalid or duplicate result rows: {name}")
        numbers = table[BUDGET_COLUMNS + ["ALC"]].to_numpy()
        if not np.isfinite(numbers).all() or not ((numbers >= 0) & (numbers <= 1)).all():
            raise ValueError(f"invalid scores: {name}")
        aligned[name] = table.set_index(KEYS).sort_index()
    reference = aligned["eb_grid"]
    for name, table in aligned.items():
        if not table.index.equals(reference.index):
            raise ValueError(f"different evaluation pairs: {name}")
        if not table[["n_eval_items", "n_acquired"]].equals(reference[["n_eval_items", "n_acquired"]]):
            raise ValueError(f"different item or acquisition counts: {name}")
    return aligned


def write_comparison(tables, run):
    aligned = align_results(tables)
    reference = aligned["eb_grid"]
    bench = reference.index.get_level_values("benchmark").to_numpy()
    unique = np.unique(bench)
    masks = [np.flatnonzero(bench == b) for b in unique]
    counts = np.array([len(m) for m in masks])
    # Reuse the same resampled benchmark clusters for every candidate.
    rng = np.random.default_rng(20261021)
    draws = rng.integers(0, len(unique), size=(2000, len(unique)))
    summary_rows, fold_rows, seed_rows, budget_rows = [], [], [], []
    for name, table in aligned.items():
        diff = table["ALC"] - reference["ALC"]
        group_sums = np.array([diff.iloc[m].sum() for m in masks])
        boots = group_sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        summary_rows.append({"method": name, "ALC": table["ALC"].mean(),
                             "delta": diff.mean(), "ci95_low": lo, "ci95_high": hi,
                             "win_fraction": (diff < -1e-9).mean()})
        for fold, values in diff.groupby(level="fold"):
            fold_rows.append({"method": name, "fold": fold, "delta": values.mean(), "rows": len(values)})
        for seed, values in diff.groupby(level="seed"):
            seed_rows.append({"method": name, "seed": seed, "ALC": table.xs(seed, level="seed")["ALC"].mean(),
                              "delta": values.mean(), "rows": len(values)})
        row = {"method": name}
        for b in BUDGET_COLUMNS:
            row[b] = table[b].mean()
            row["delta_" + b] = (table[b] - reference[b]).mean()
        budget_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("ALC")
    by_fold, by_seed, budgets = pd.DataFrame(fold_rows), pd.DataFrame(seed_rows), pd.DataFrame(budget_rows)
    for filename, table in [("summary.csv", summary), ("by_fold.csv", by_fold),
                             ("by_seed.csv", by_seed), ("by_budget.csv", budgets)]:
        table.to_csv(run / filename, index=False)
    distinct_pairs = reference.reset_index().drop_duplicates(["benchmark", "subject_key"])
    parts = [f"評価行数: {len(reference)} / 重複を除くペア: {len(distinct_pairs)} / ベンチマーク: {len(unique)}",
             "deltaは候補 - eb_grid。負が改善。全予算を固定した同一設定で比較。",
             summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             "\nseedごとのALCと差（seed=0は従来設定の再現確認用）:",
             by_seed.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             "\nfoldごとの差（seedをまとめた平均）:",
             by_fold.pivot(index="method", columns="fold", values="delta").to_string(float_format=lambda x: f"{x:+.6f}"),
             "\n予算ごとのBrier:",
             budgets[["method"] + BUDGET_COLUMNS].to_string(index=False, float_format=lambda x: f"{x:.6f}"),
             "\n95%区間はベンチマーク単位の探索的ブートストラップ。多重比較や候補選択の偏りは未補正。",
             "seedを変えてもベンチマークfoldは固定。同じペアの再出現は独立した新規データではありません。",
             "追加seedは抽出と獲得ラベルに対する安定性の確認であり、未使用ベンチマークでの最終評価ではありません。",
             "既に同じfoldを使って開発しているため、未知データへの改善はこの結果だけでは確定しません。"]
    text = "\n".join(parts)
    (run / "comparison.txt").write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)


def run(args):
    root = Path(args.root).resolve()
    work = root / "work"
    package = Path(__file__).resolve().parent
    reference_path = Path(args.reference).resolve() if args.reference else root / "results/eb_grid.json"
    required = [reference_path, root / "train_eb_prior.py", root / "submissions/eb_grid/model.py",
                *(work / n for n in ["folds.json", "responses.parquet", "items.parquet", "subjects.parquet"])]
    if any(not p.is_file() for p in required):
        raise ValueError("missing files:\n" + "\n".join(str(p) for p in required if not p.is_file()))
    if (root / "submissions/eb_grid/labeling.py").exists():
        raise ValueError("baseline has labeling.py; this experiment assumes random acquisition")
    reference = json.loads(reference_path.read_text())
    fold_data = json.loads((work / "folds.json").read_text())
    settings = validate_settings(reference, fold_data, args.seeds, args.n_draws)
    folds = fold_data["folds"]
    configs = configurations(settings["draw_limit"])
    if args.methods:
        if len(set(args.methods)) != len(args.methods) or set(args.methods) - set(configs):
            raise ValueError("unknown or duplicate methods: " + " ".join(args.methods))
        configs = {name: config for name, config in configs.items() if name == "eb_grid" or name in args.methods}
    (root / "results").mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="eb_ablation_", dir=root / "results"))
    print(f"保存先: {run_dir}\nGPUは使用しません。主な計算ライブラリのスレッド設定は4です。", flush=True)
    print("設定: " + json.dumps(settings), flush=True)
    code = run_dir / "code"
    code.mkdir()
    for source in package.iterdir():
        if source.suffix in (".py", ".txt"):
            shutil.copy2(source, code / source.name)
    shutil.copy2(root / "train_eb_prior.py", code / "reference_train_eb_prior.py")
    shutil.copy2(root / "submissions/eb_grid/model.py", code / "reference_model.py")
    shutil.copy2(work / "folds.json", run_dir / "folds.json")
    identity = data_identity(work)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "root": str(root), "settings": settings, "candidates": configs,
                "data_identity": identity, "reference_result": reference,
                "reference_result_path": str(reference_path),
                "source_sha256": {p.name: sha256(p) for p in code.iterdir() if p.is_file()},
                "selection_note": "Predefined 2x2x2 ablation plus Beta(1,1); no automatic winner adoption."}
    dump(run_dir / "manifest.json", manifest)
    stats = collect_stats(work, folds, run_dir)
    manifest["pair_stats_sha256"] = sha256(run_dir / "pair_stats.csv")
    dump(run_dir / "manifest.json", manifest)
    tables = {name: [] for name in configs}
    prior_rows = []
    for fold in range(settings["n_folds"]):
        priors = {cap: fit_prior(stats, folds, fold, cap)
                  for cap in dict.fromkeys(config["max_items"] for config in configs.values())}
        if fold == 0:
            verify_original_training(code, work, priors[None], run_dir)
        for cap, prior in priors.items():
            assert not set(prior["training_benchmarks"]) & {b for b, f in folds.items() if f == fold}
            prior_rows.append({"fold": fold, "max_items": "none" if cap is None else cap,
                               **{key: prior[key] for key in ["n_pairs", "n_benchmarks", "mean", "tail0", "tail1"]}})
        models = {}
        for name, config in configs.items():
            directory = run_dir / "models" / name / f"fold_{fold}"
            directory.mkdir(parents=True)
            dump(directory / "prior.json", priors[config["max_items"]])
            dump(directory / "config.json", config)
            shutil.copy2(code / ("reference_model.py" if name == "eb_grid" else "model.py"), directory / "model.py")
            if name != "eb_grid":
                shutil.copy2(code / "eb_core.py", directory / "eb_core.py")
            models[name] = evaluator.load_submission(str(directory))
        held_out = sorted(b for b, f in folds.items() if f == fold)
        for seed in settings["snapshot_seeds"]:
            start = time.monotonic()
            snapshot, groups = evaluator.build_snapshot(str(work), held_out, settings["n_draws"],
                                                        settings["draw_limit"], seed, fold)
            print(f"fold={fold} seed={seed}: {len(snapshot)} pairs, 準備 {time.monotonic()-start:.1f}s", flush=True)
            if not snapshot:
                continue
            for name, (directory, model, labeling) in models.items():
                acquired, scores = evaluator.evaluate(snapshot, model, labeling, directory, groups)
                table = evaluator.summarize(snapshot, acquired, scores)
                table.insert(0, "fold", fold)
                table.insert(0, "seed", seed)
                tables[name].append(table)
                # Keep finished folds if a later fold fails.
                pd.concat(tables[name], ignore_index=True).to_csv(run_dir / f"{name}.csv", index=False)
                print(f"  {name}: ALC={table['ALC'].mean():.6f}", flush=True)
            del snapshot
    if not all(tables.values()):
        raise ValueError("no evaluation pairs")
    if data_identity(work) != identity:
        raise ValueError("data or folds changed during the run; do not use this comparison")
    complete = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    pd.DataFrame(prior_rows).to_csv(run_dir / "prior_diagnostics.csv", index=False)
    write_comparison(complete, run_dir)
    manifest["status"] = "complete"
    dump(run_dir / "manifest.json", manifest)
    print(f"\n共有する結果: {run_dir / 'comparison.txt'}", flush=True)
    print(f"事前分布の診断: {run_dir / 'prior_diagnostics.csv'}", flush=True)
    print("models内はCV fold用です。提出ZIPとしては使わないでください。", flush=True)
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--reference", help="reference result JSON; default: ROOT/results/eb_grid.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5], help="snapshot seeds; fold assignment stays fixed")
    ap.add_argument("--n-draws", type=int, help="default: reference JSON's n_draws")
    ap.add_argument("--methods", nargs="+", help="optional candidate subset; eb_grid is always included")
    args = ap.parse_args()
    try:
        # Arrow has its own pool, separate from BLAS/OMP.
        import pyarrow as pa
        pa.set_cpu_count(4)
        pa.set_io_thread_count(2)
        run(args)
    except (ValueError, KeyError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
