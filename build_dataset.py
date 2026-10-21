"""公開学習データ（payload/data/*/formatted_tables）を 3 つの表にまとめる。

出力（--out、既定 work/）:
  responses.parquet  1 行 = 1 回の応答（予測対象の単位）
      benchmark_id, subject_key, item_id, trial, test_condition, interactors, response
  subjects.parquet   1 行 = 1 subject（評価時に predict() が受け取る 8 項目と同じ列）
      subject_key, normalized_name, provider, release_date, access_date,
      harness, harness_version, reasoning_effort, subject_features_extra
  items.parquet      1 行 = 1 問題
      item_id, benchmark_id, item_content, item_features
  summary.csv        ベンチマークごとの集計

subject_key について:
  評価時の predict() には subject_id が渡されず、subject の属性 8 項目だけが渡される。
  そのため、学習時も同じ 8 項目（欠損は空文字）を連結したものを subject の識別子とする。
  これにより、ベンチマークをまたいで同じ AI システムを同一視できる。

使い方:
    python build_dataset.py --data payload/data --out work
"""

import argparse
import glob
import hashlib
import json
import os

import pandas as pd

# 評価時に input[0]（subject）に入る項目。順番も公式ページと同じ。
SUBJECT_FIELDS = [
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "harness_version",
    "reasoning_effort",
    "subject_features_extra",
]


def _as_str(series: pd.Series) -> pd.Series:
    """評価時と同じく、すべて文字列・欠損は空文字にそろえる。"""
    return series.where(series.notna(), "").astype(str)


def subject_key(row) -> str:
    joined = "\x1f".join(row[f] for f in SUBJECT_FIELDS)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def load_benchmark(table_dir: str):
    meta = pd.read_parquet(os.path.join(table_dir, "benchmarks.parquet")).iloc[0]
    subjects = pd.read_parquet(os.path.join(table_dir, "subjects.parquet"))
    items = pd.read_parquet(os.path.join(table_dir, "items.parquet"))
    resp = pd.read_parquet(os.path.join(table_dir, "response.parquet"))

    for f in SUBJECT_FIELDS:
        if f not in subjects.columns:
            subjects[f] = ""
        subjects[f] = _as_str(subjects[f])
    subjects["subject_key"] = subjects.apply(subject_key, axis=1)

    resp = resp.merge(
        subjects[["subject_id", "subject_key"]], on="subject_id", how="left"
    )
    missing = resp["subject_key"].isna().sum()
    if missing:
        print(f"  warning: {meta['benchmark_id']}: {missing} responses without subject")
        resp = resp[resp["subject_key"].notna()]

    for col in ["test_condition", "interactors"]:
        if col not in resp.columns:
            resp[col] = ""
        resp[col] = _as_str(resp[col])

    items = items.rename(columns={"content": "item_content"})
    for col in ["item_content", "item_features"]:
        if col not in items.columns:
            items[col] = ""
        items[col] = _as_str(items[col])

    return meta, subjects, items, resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="payload/data")
    ap.add_argument("--out", default="work")
    args = ap.parse_args()
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit("出力先が空ではありません。新しい出力先を指定してください")
    os.makedirs(args.out, exist_ok=True)

    table_dirs = sorted(
        os.path.dirname(p)
        for p in glob.glob(os.path.join(args.data, "*", "**", "benchmarks.parquet"), recursive=True)
    )
    print(f"{len(table_dirs)} benchmarks")

    all_resp, all_subj, all_items, rows = [], [], [], []
    for d in table_dirs:
        meta, subjects, items, resp = load_benchmark(d)
        bid = str(meta["benchmark_id"])

        values = set(resp["response"].dropna().unique().tolist())
        if not values <= {0.0, 1.0}:
            print(f"  skip {bid}: non-binary responses {sorted(values)[:5]}")
            continue
        resp = resp[resp["response"].notna()].copy()
        if resp.empty:
            print(f"  skip {bid}: no usable responses")
            continue
        # trial が欠損しているベンチマークは 1 回試行として扱う
        n_missing_trial = int(resp["trial"].isna().sum())
        if n_missing_trial:
            print(f"  note {bid}: {n_missing_trial} responses without trial -> set to 1")
        resp["trial"] = resp["trial"].fillna(1).astype("int64")

        all_resp.append(
            resp[["benchmark_id", "subject_key", "item_id", "trial",
                  "test_condition", "interactors", "response"]]
        )
        all_subj.append(subjects[["subject_key"] + SUBJECT_FIELDS])
        all_items.append(items[["item_id", "benchmark_id", "item_content", "item_features"]])
        rows.append({
            "benchmark_id": bid,
            "domain": json.dumps(list(meta["domain"])) if meta["domain"] is not None else "",
            "n_subjects": resp["subject_key"].nunique(),
            "n_items": resp["item_id"].nunique(),
            "n_responses": len(resp),
            "mean_response": resp["response"].mean(),
            "max_trial": int(resp["trial"].max()),
        })

    responses = pd.concat(all_resp, ignore_index=True)
    responses["response"] = responses["response"].astype("int8")
    subjects = pd.concat(all_subj, ignore_index=True).drop_duplicates("subject_key")
    items = pd.concat(all_items, ignore_index=True).drop_duplicates(["benchmark_id", "item_id"])
    summary = pd.DataFrame(rows).sort_values("n_responses", ascending=False)

    responses.to_parquet(os.path.join(args.out, "responses.parquet"), index=False)
    subjects.to_parquet(os.path.join(args.out, "subjects.parquet"), index=False)
    items.to_parquet(os.path.join(args.out, "items.parquet"), index=False)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    # 概要の表示
    per_subject = responses.groupby("subject_key")["benchmark_id"].nunique()
    print("\n===== 概要 =====")
    print(f"ベンチマーク数      : {summary.shape[0]}")
    print(f"応答数              : {len(responses):,}")
    print(f"subject 数          : {subjects.shape[0]:,}")
    print(f"問題数              : {items.shape[0]:,}")
    print(f"全体の正答率        : {responses['response'].mean():.3f}")
    print("1 subject あたりの出現ベンチマーク数:")
    print(per_subject.describe().round(2).to_string())
    print("\n応答数の多いベンチマーク（上位 15）:")
    print(summary.head(15).to_string(index=False))
    print(f"\n保存先: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
