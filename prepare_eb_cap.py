#!/usr/bin/env python3
"""Train the selected eb_cap on all folds, validate it, and create a submission ZIP.

Run in the existing server container:
    python -u prepare_eb_cap.py --root /workspace/paiec-eb-cap

Requires the included experiments package and a separately obtained starting kit.
Uses no GPU. Does not upload, edit existing submissions, or change folds.
"""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def identity(work):
    st = (work / "responses.parquet").stat()
    return {"responses_size": st.st_size, "responses_mtime_ns": st.st_mtime_ns,
            "folds_sha256": sha(work / "folds.json"), "summary_sha256": sha(work / "summary.csv")}


def checked_command(command, log_path, cwd):
    with Path(log_path).open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if proc.wait():
                raise RuntimeError(f"公式チェックに失敗しました。提出せずログを確認してください: {log_path}")


def build(root):
    root = Path(root).resolve()
    work = root / "work"
    package = root / "experiments"
    checks = root / "starting_kit/tools"
    required = [work / "folds.json", work / "summary.csv", work / "responses.parquet",
                root / "results/eb_grid.json", root / "submissions/eb_grid/model.py",
                package / "pair_stats.py", package / "eb_core.py",
                checks / "check_submission_zip.py", checks / "run_smoke_test.py"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError("必要ファイルがありません:\n" + "\n".join(missing))
    folds = json.loads((work / "folds.json").read_text())["folds"]
    summary = pd.read_csv(work / "summary.csv", dtype={"benchmark_id": str})
    if not folds or set(summary["benchmark_id"]) != set(folds):
        raise ValueError("folds.jsonとsummary.csvのベンチマーク一覧が一致しません")
    reference = json.loads((root / "results/eb_grid.json").read_text())
    if int(reference["args"]["draw_limit"]) != 1000:
        raise ValueError("元の評価設定がdraw_limit=1000ではありません。実験条件を確認してください")
    if (root / "submissions/eb_grid/labeling.py").exists():
        raise ValueError("元モデルにlabeling.pyが追加されています。今回の実験条件と異なります")
    initial_identity = identity(work)
    (root / "dist").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="eb_cap_all_", dir=root / "dist"))
    code, submission = output / "code", output / "submission"
    code.mkdir()
    submission.mkdir()
    print(f"作成先: {output}", flush=True)
    for name in ["pair_stats.py", "eb_core.py"]:
        shutil.copy2(package / name, code / name)
    shutil.copy2(__file__, code / "prepare_eb_cap.py")
    shutil.copy2(root / "submissions/eb_grid/model.py", submission / "model.py")
    shutil.copy2(work / "folds.json", output / "folds.json")
    stats_code = load_module(code / "pair_stats.py", "cap_training")
    inference_code = load_module(code / "eb_core.py", "cap_reference_inference")
    manifest = {"status": "training", "created_at_jst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
                "method": "eb_cap", "training_fold": "all", "max_items": 1000,
                "uniform_mix": 0.0, "zero_half": False,
                "data_identity": initial_identity,
                "source_sha256": {p.name: sha(p) for p in code.iterdir() if p.is_file()},
                "model_sha256": sha(submission / "model.py"),
                "ai_assistance": "Codex: experiment implementation, validation and packaging; user executes on server."}
    dump(output / "manifest.json", manifest)
    tables = []
    with (output / "training.log").open("w", encoding="utf-8") as log:
        for i, bid in enumerate(sorted(folds), 1):
            responses = pd.read_parquet(work / "responses.parquet",
                                        columns=["benchmark_id", "subject_key", "item_id", "response"],
                                        filters=[("benchmark_id", "in", [bid])])
            table = stats_code.aggregate_benchmark(responses, bid)
            # Filter early, after the full per-item aggregation, to retain only training pairs.
            table = table[table["n_items"].between(80, 1000)]
            tables.append(table)
            line = f"集計 {i}/{len(folds)}: {bid}, 採用ペア={len(table)}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            del responses
        stats = pd.concat(tables, ignore_index=True)
        prior = stats_code.fit_prior(stats, folds, "all", max_items=1000)
        if identity(work) != initial_identity:
            raise ValueError("学習中にデータまたはfold一覧が変更されました")
        if prior["fold"] != "all" or prior["max_items"] != 1000:
            raise ValueError("incorrect training scope")
        expected_benchmarks = sorted(stats["benchmark_id"].unique())
        if prior["training_benchmarks"] != expected_benchmarks or prior["n_pairs"] != len(stats):
            raise ValueError("training inventory mismatch")
        dump(submission / "prior.json", prior)
        line = (f"全fold学習完了: benchmarks={prior['n_benchmarks']} pairs={prior['n_pairs']} "
                f"prior_mean={prior['mean']:.6f}, max_items=1000, fold=all")
        print(line, flush=True)
        log.write(line + "\n")
    model = load_module(submission / "model.py", "cap_submission")
    expected = inference_code.Predictor(prior)
    subject = {"normalized_name": "local_test", "provider": "", "release_date": "", "access_date": "",
               "harness": "one", "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
    item = {"benchmark_id": "benchmark_local", "item_content": "test", "item_features": "", "interactors": ""}
    foreign = [[[{**subject, "harness": "two"}, item], 1],
               [[subject, {**item, "benchmark_id": "benchmark_other"}], 1]]
    for n in range(32):
        for s in range(n + 1):
            labeled = foreign + [[[subject, item], 1]] * s + [[[subject, item], 0]] * (n - s)
            p = float(model.predict([subject, item], labeled))
            if not math.isfinite(p) or not 0 <= p <= 1 or not math.isclose(p, expected.posterior_mean(s, n), abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"CVモデルとの予測不一致: successes={s}, labels={n}")
    print("予測一致検査: 528通りすべてPASS", flush=True)
    # ZIP contains only these two files at its root, without caches or training code.
    pending = output / "candidate.zip"
    members = ["model.py", "prior.json"]
    with ZipFile(pending, "w", ZIP_DEFLATED) as archive:
        for member in members:
            archive.write(submission / member, arcname=member)
    with ZipFile(pending) as archive:
        if archive.testzip() is not None or sorted(archive.namelist()) != sorted(members):
            raise ValueError("invalid archive")
        for member in members:
            if archive.read(member) != (submission / member).read_bytes():
                raise ValueError("archive contents differ")
    manifest.update(status="checking", n_pairs=prior["n_pairs"], n_benchmarks=prior["n_benchmarks"],
                    prior_mean=prior["mean"], inference_cases_checked=528)
    dump(output / "manifest.json", manifest)
    checked_command([sys.executable, str(checks / "check_submission_zip.py"), str(pending)],
                    output / "zip_check.log", root)
    checked_command([sys.executable, "-u", str(checks / "run_smoke_test.py"), str(submission)],
                    output / "smoke_test.log", root)
    with ZipFile(pending) as archive:
        if any(archive.read(member) != (submission / member).read_bytes() for member in members):
            raise ValueError("モデルまたはpriorがチェック中に変更されました")
    final = output / "eb_cap_all.zip"
    pending.rename(final)
    manifest.update(status="checks_passed", zip_name=final.name, zip_sha256=sha(final),
                    members_sha256={member: sha(submission / member) for member in members})
    dump(output / "manifest.json", manifest)
    (output / "CHECKS_PASSED.txt").write_text("予測一致検査、公式ZIP検査、公式スモークテストが正常終了しました。\n", encoding="utf-8")
    print(f"\n提出準備完了: {final}\nSHA256: {manifest['zip_sha256']}\n自動送信は行っていません。", flush=True)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    import pyarrow as pa
    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    build(args.root)


if __name__ == "__main__":
    main()
