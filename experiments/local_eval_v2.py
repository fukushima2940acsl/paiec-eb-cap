"""公式仕様を参考に、提出用の model.py を手元で採点するローカル評価器。

非公開の本番環境との完全な一致は保証しない。既知の簡略化は下記を参照。

再現対象の仕様（https://aims.stanford.edu/competition）:
  - ベンチマーク単位のコールドスタート：検証用ベンチマークは学習に使わない（fold 分割）。
  - 80 問以上ある subject×benchmark ペアだけを対象にし、問題を 50/50 で獲得用・評価用に分ける。
    分割は seed で固定され、提出（手法）が変わっても同じ。
  - ラベル予算 0, 1, 3, 7, 15, 31。獲得は入れ子（小さい予算のラベルは大きい予算でも使える）。
  - 1 回の評価（draw）では、ペアを丸ごと、元の問題数の合計が 1,000 以下になるように選ぶ
    （Starting Kit の README）。その回でまだ少ないベンチマーク・subject を優先する。
    結果として 1 回あたり 8 ペア前後、ほぼ別々のベンチマークになる。1,000 問を超えるペアは選ばれない。
  - labeled には、同じ評価回に抽出されたペアの獲得済みラベルが入る（[[subject, item], label] の形）。
  - 奇数の問題数なら獲得用が 1 問多い。
  - 評価用の問題ごとに predict を 1 回呼び、その問題の全応答（複数試行）で Brier を計算する。
  - Brier ALC = 0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31 をペアごとに計算し、ペアを均等に平均する。
  - labeling.py があれば acquisition_function で獲得する問題を選ぶ。なければランダム（獲得用の並び順の先頭から）。

本番との違い（計算時間のための簡略化）:
  - fold ごとの評価回の数を --n-draws で決める（同じペアは複数の評価回に入れない）。
  - 匿名の benchmark_id は、ベンチマーク名のハッシュで作る。
  - labeling.py の獲得関数に渡す labeled は、そのペア自身の獲得済みラベルだけ。

使い方:
    python tools/reproduce_cv.py --phase confirm

学習が必要な手法は、fold ごとの学習データ（--fold k の「検証用以外」のベンチマーク）で
学習した成果物を submission ディレクトリに置いてから評価すること。
fold の割り当ては work/folds.json に保存され、学習スクリプトからも同じものを使う。
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

BUDGETS = [0, 1, 3, 7, 15, 31]
ALC_WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
MAX_LABELS = 31
MIN_ITEMS = 80

SUBJECT_FIELDS = [
    "normalized_name", "provider", "release_date", "access_date",
    "harness", "harness_version", "reasoning_effort", "subject_features_extra",
]


# ---------------------------------------------------------------------------
# 共通のユーティリティ（学習スクリプトからも import して使う）
# ---------------------------------------------------------------------------

def stable_seed(*parts) -> int:
    text = "\x1f".join(str(p) for p in parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def anon_benchmark_id(name: str) -> str:
    return "benchmark_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]


def load_folds(work: str, n_folds: int = 5, seed: int = 0) -> dict:
    """benchmark_id -> fold 番号。初回に作って work/folds.json に保存し、以後は再利用する。"""
    path = os.path.join(work, "folds.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if data["n_folds"] == n_folds and data["seed"] == seed:
            return data["folds"]
    summary = pd.read_csv(os.path.join(work, "summary.csv"))
    names = sorted(summary["benchmark_id"].astype(str))
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    folds = {name: i % n_folds for i, name in enumerate(names)}
    with open(path, "w") as f:
        json.dump({"n_folds": n_folds, "seed": seed, "folds": folds}, f, indent=2)
    return folds


def brier_alc(per_budget: dict) -> float:
    return float(sum(ALC_WEIGHTS[b] * per_budget[b] for b in BUDGETS))


# ---------------------------------------------------------------------------
# 提出物の読み込み
# ---------------------------------------------------------------------------

def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_submission(directory: str):
    directory = os.path.abspath(directory)
    sys.path.insert(0, directory)  # 提出物内の補助モジュールを import できるように
    cwd = os.getcwd()
    os.chdir(directory)  # 本番と同じく、提出物のルートを作業ディレクトリにする
    try:
        model = load_module(os.path.join(directory, "model.py"), "submission_model")
        labeling = None
        labeling_path = os.path.join(directory, "labeling.py")
        if os.path.exists(labeling_path):
            labeling = load_module(labeling_path, "submission_labeling")
    finally:
        os.chdir(cwd)
    return directory, model, labeling


# ---------------------------------------------------------------------------
# 評価データ（スナップショット）の構築
# ---------------------------------------------------------------------------

def draw_groups(counts, n_draws, draw_limit, seed, fold):
    """本番の抽出を再現する。

    Starting Kit の説明：1 回の評価では、(benchmark, subject) のペアを丸ごと、
    元の問題数の合計が draw_limit（1,000）以下になるように選ぶ。
    その評価回でまだ少ないベンチマーク・subject を優先し、同順位はランダム。
    入りきらないペアは飛ばし、もう入るペアがなくなったら終了。
    ここでは評価回（draw）を n_draws 回作る。ペアは評価回をまたいで重複させない。
    """
    pool = [(bid, skey, int(n)) for (bid, skey), n in counts.items()
            if MIN_ITEMS <= n <= draw_limit]
    rng = np.random.default_rng(stable_seed(seed, "draw", fold))
    rng.shuffle(pool)
    used = set()
    draws = []
    for _ in range(n_draws):
        remaining = draw_limit
        bench_n, subj_n, chosen = {}, {}, []
        while True:
            best, best_key = None, None
            for idx, (bid, skey, n) in enumerate(pool):
                if idx in used or n > remaining:
                    continue
                key = (bench_n.get(bid, 0), subj_n.get(skey, 0), idx)
                if best_key is None or key < best_key:
                    best, best_key = idx, key
            if best is None:
                break
            bid, skey, n = pool[best]
            used.add(best)
            chosen.append((bid, skey))
            remaining -= n
            bench_n[bid] = bench_n.get(bid, 0) + 1
            subj_n[skey] = subj_n.get(skey, 0) + 1
        if not chosen:
            break
        draws.append(chosen)
    return draws


def build_snapshot(work, held_out, n_draws, draw_limit, seed, fold):
    """検証用ベンチマークから評価回（draw）を作り、各ペアの獲得用・評価用の問題を用意する。
    戻り値: (ペアのリスト, 評価回ごとのペア番号のリスト)"""
    responses = pd.read_parquet(
        os.path.join(work, "responses.parquet"),
        filters=[("benchmark_id", "in", list(held_out))],
    )
    subjects = pd.read_parquet(os.path.join(work, "subjects.parquet")).set_index("subject_key")
    items = pd.read_parquet(
        os.path.join(work, "items.parquet"),
        filters=[("benchmark_id", "in", list(held_out))],
    ).drop_duplicates(["benchmark_id", "item_id"]).set_index(["benchmark_id", "item_id"])

    counts = responses.groupby(["benchmark_id", "subject_key"])["item_id"].nunique()
    draws = draw_groups(counts, n_draws, draw_limit, seed, fold)

    grouped = responses.set_index(["benchmark_id", "subject_key"]).sort_index()
    snapshot, groups = [], []
    for draw in draws:
        group = []
        for bid, skey in draw:
            group.append(len(snapshot))
            rows = grouped.loc[(bid, skey)]
            srow = subjects.loc[skey]
            subject = {f: str(srow[f]) for f in SUBJECT_FIELDS}

            # 問題を固定の乱数で 50/50 に分ける（奇数なら獲得用が 1 問多い）
            item_ids = sorted(rows["item_id"].unique())
            rng = np.random.default_rng(stable_seed(seed, "split", bid, skey))
            rng.shuffle(item_ids)
            half = (len(item_ids) + 1) // 2
            acq_ids, eval_ids = item_ids[:half], item_ids[half:]

            by_item = rows.groupby("item_id")

            def make_item(iid):
                irow = items.loc[(bid, iid)]
                first = by_item.get_group(iid).iloc[0]
                return {
                    "item_content": str(irow["item_content"]),
                    "item_features": str(irow["item_features"]),
                    "interactors": str(first["interactors"]),
                    "benchmark_id": anon_benchmark_id(bid),
                }

            acquisition = []
            for iid in acq_ids:
                labels = by_item.get_group(iid)["response"].astype(int).tolist()
                acquisition.append((make_item(iid), labels[0]))  # 1 件の記録済み応答を明かす
            evaluation = []
            for iid in eval_ids:
                labels = by_item.get_group(iid)["response"].astype(int).tolist()
                evaluation.append((make_item(iid), labels))

            snapshot.append({
                "benchmark": bid,
                "subject_key": skey,
                "subject_id": "subject_" + skey[:6],
                "subject": subject,
                "acquisition": acquisition,
                "evaluation": evaluation,
            })
        groups.append(group)
    return snapshot, groups


# ---------------------------------------------------------------------------
# 獲得と採点
# ---------------------------------------------------------------------------

def acquire(pair, model, labeling):
    """このペアで明かすラベルの列を決める（入れ子なので先頭 n 件が予算 n のラベル）。"""
    subject = pair["subject"]
    if labeling is None:
        return [[[subject, item], label] for item, label in pair["acquisition"][:MAX_LABELS]]

    acquired = []
    candidates = pair["acquisition"]
    for idx, (item, label) in enumerate(candidates):
        if len(acquired) >= MAX_LABELS:
            break
        inp = [subject, item]
        prediction = float(model.predict(inp, list(acquired)))
        context = {
            "subject_id": pair["subject_id"],
            "benchmark_id": item["benchmark_id"],
            "labels_acquired": len(acquired),
            "labels_remaining": MAX_LABELS - len(acquired),
            "max_labels": MAX_LABELS,
            "items_remaining": len(candidates) - idx,
        }
        if labeling.acquisition_function(inp, prediction, list(acquired), context):
            acquired.append([inp, label])
    return acquired


def check_probability(p) -> float:
    p = float(p)
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"invalid probability: {p}")
    return p


def evaluate(snapshot, model, labeling, submission_dir, groups):
    cwd = os.getcwd()
    os.chdir(submission_dir)
    try:
        acquired = [acquire(pair, model, labeling) for pair in snapshot]
        results = {i: {} for i in range(len(snapshot))}
        for budget in BUDGETS:
            for group in groups:
                # labeled には、同じグループ（同じ評価回）のペアのラベルだけが入る
                labeled = [entry for i in group for entry in acquired[i][:budget]]
                for i in group:
                    pair = snapshot[i]
                    sq_errors = []
                    for item, labels in pair["evaluation"]:
                        p = check_probability(model.predict([pair["subject"], item], labeled))
                        sq_errors.extend((p - y) ** 2 for y in labels)
                    results[i][budget] = float(np.mean(sq_errors))
    finally:
        os.chdir(cwd)
    return acquired, results


def summarize(snapshot, acquired, results):
    rows = []
    for i, pair in enumerate(snapshot):
        row = {
            "benchmark": pair["benchmark"],
            "subject_key": pair["subject_key"],
            "subject": pair["subject"]["normalized_name"] or pair["subject_key"][:8],
            "n_eval_items": len(pair["evaluation"]),
            "n_acquired": len(acquired[i]),
        }
        row.update({f"B{b}": results[i][b] for b in BUDGETS})
        row["ALC"] = brier_alc(results[i])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="model.py を含むディレクトリ")
    ap.add_argument("--work", default="work")
    ap.add_argument("--fold", default="0", help="検証に使う fold 番号、または all")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-draws", type=int, default=12, help="fold ごとに作る評価回の数")
    ap.add_argument("--draw-limit", type=int, default=1000, help="1 回の評価で抽出する元の問題数の上限")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="結果の保存先（.json）。ペアごとの結果は同名の .csv")
    ap.add_argument(
        "--train-cmd", default=None,
        help="各 fold の評価前に実行する学習コマンド。{fold} {submission} {work} が置き換わる",
    )
    args = ap.parse_args()

    work = os.path.abspath(args.work)
    folds = load_folds(work, args.n_folds, args.seed)
    fold_list = list(range(args.n_folds)) if args.fold == "all" else [int(args.fold)]

    tables = []
    for k in fold_list:
        if args.train_cmd:
            # fold k を除いたベンチマークで学習し、成果物を submission に書き出す
            cmd = args.train_cmd.format(fold=k, submission=args.submission, work=args.work)
            print(f"fold {k}: 学習 $ {cmd}")
            if subprocess.run(cmd, shell=True).returncode != 0:
                raise SystemExit(f"fold {k}: 学習コマンドが失敗しました")
        # 学習結果を読み直すため、fold ごとに提出物を読み込み直す
        submission_dir, model, labeling = load_submission(args.submission)
        if k == fold_list[0]:
            print(f"submission: {submission_dir}  (labeling.py: {'あり' if labeling else 'なし'})")

        held_out = sorted(b for b, f in folds.items() if f == k)
        t0 = time.time()
        snapshot, groups = build_snapshot(
            work, held_out, args.n_draws, args.draw_limit, args.seed, k
        )
        if not snapshot:
            print(f"fold {k}: 評価できるペアがない（80〜{args.draw_limit} 問のペアなし）")
            continue
        t1 = time.time()
        acquired, results = evaluate(snapshot, model, labeling, submission_dir, groups)
        table = summarize(snapshot, acquired, results)
        table.insert(0, "fold", k)
        tables.append(table)
        print(
            f"fold {k}: benchmarks={table['benchmark'].nunique()} pairs={len(table)} "
            f"ALC={table['ALC'].mean():.4f}  (準備 {t1 - t0:.0f}s, 評価 {time.time() - t1:.0f}s)"
        )

    if not tables:
        return
    table = pd.concat(tables, ignore_index=True)
    budget_means = {f"B{b}": float(table[f"B{b}"].mean()) for b in BUDGETS}
    overall = float(table["ALC"].mean())

    print("\n===== 結果 =====")
    print(f"Brier ALC（ペア平均）: {overall:.4f}   pairs={len(table)}")
    print("予算ごとの Brier: " + "  ".join(f"{k}={v:.4f}" for k, v in budget_means.items()))
    by_bench = table.groupby("benchmark")["ALC"].agg(["mean", "count"]).sort_values("mean")
    print("\nベンチマークごとの ALC:")
    print(by_bench.round(4).to_string())

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "submission": submission_dir,
                "args": vars(args),
                "brier_alc": overall,
                "budgets": budget_means,
                "n_pairs": len(table),
            }, f, indent=2, ensure_ascii=False)
        table.to_csv(os.path.splitext(args.out)[0] + ".csv", index=False)
        print(f"\n保存: {args.out}")


if __name__ == "__main__":
    main()
