"""経験ベイズ用の事前分布（ペアの正答率の分布）を学習データから推定する。

学習用ベンチマーク（--fold k を指定したら fold k 以外。all なら全ベンチマーク）について、
80 問以上ある subject×benchmark ペアの正答率を求め、
[0, 1] を等間隔に区切ったグリッド上のヒストグラムとして保存する。

  - 各ベンチマークの重みを等しくする（ペア数の多いベンチマークが分布を支配しないように）。
  - 正答率 0 や 1 のペア（全問正解・全問不正解型）も、そのままグリッドの端に入る。

使い方:
    python train_eb_prior.py --fold 0 --out submissions/eb_grid/prior.json
    python train_eb_prior.py --fold all --out submissions/eb_grid/prior.json   # 提出用
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

MIN_ITEMS = 80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--fold", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=101, help="グリッドの点の数")
    ap.add_argument("--smooth", type=float, default=1e-3, help="空のビンに足す小さな重み")
    args = ap.parse_args()

    with open(os.path.join(args.work, "folds.json")) as f:
        folds = json.load(f)["folds"]
    if args.fold == "all":
        train = sorted(folds)
    else:
        train = sorted(b for b, k in folds.items() if k != int(args.fold))

    resp = pd.read_parquet(
        os.path.join(args.work, "responses.parquet"),
        columns=["benchmark_id", "subject_key", "item_id", "response"],
        filters=[("benchmark_id", "in", train)],
    )
    # 問題ごとの正答率（複数試行は平均）→ ペアごとの正答率
    per_item = resp.groupby(["benchmark_id", "subject_key", "item_id"])["response"].mean()
    per_pair = per_item.groupby(["benchmark_id", "subject_key"]).agg(["mean", "size"])
    per_pair = per_pair[per_pair["size"] >= MIN_ITEMS].reset_index()

    grid = np.linspace(0.0, 1.0, args.grid)
    weights = np.zeros(args.grid)
    n_bench = 0
    for _, group in per_pair.groupby("benchmark_id"):
        idx = np.rint(group["mean"].to_numpy() * (args.grid - 1)).astype(int)
        hist = np.bincount(idx, minlength=args.grid).astype(float)
        weights += hist / hist.sum()  # ベンチマークごとに合計 1
        n_bench += 1
    weights += args.smooth
    weights /= weights.sum()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "grid": grid.round(6).tolist(),
            "weights": weights.tolist(),
            "fold": args.fold,
            "n_benchmarks": n_bench,
            "n_pairs": int(len(per_pair)),
        }, f)

    mean = float((grid * weights).sum())
    tail0 = float(weights[grid <= 0.02].sum())
    tail1 = float(weights[grid >= 0.98].sum())
    print(
        f"prior: benchmarks={n_bench} pairs={len(per_pair)} mean={mean:.3f} "
        f"P(p<=0.02)={tail0:.3f} P(p>=0.98)={tail1:.3f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
