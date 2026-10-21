"""Connect an existing prepared dataset and Starting Kit without committing them."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from release_utils import ROOT, link_directory, save_once


def connect(work, starting_kit, root=ROOT):
    work, root = Path(work).resolve(strict=True), Path(root)
    folds = json.loads((root / 'configs/folds.json').read_text())
    summary = pd.read_csv(work / 'summary.csv', dtype={'benchmark_id': str})
    if set(summary['benchmark_id']) != set(folds['folds']):
        raise ValueError('データのベンチマーク一覧と公開版のfolds.jsonが一致しません')
    for name in ['responses.parquet', 'items.parquet', 'subjects.parquet']:
        if not (work / name).is_file():
            raise ValueError(f'必要なデータがありません: {name}')
    kit = Path(starting_kit).resolve(strict=True)
    for name in ['check_submission_zip.py', 'run_smoke_test.py']:
        if not (kit / 'tools' / name).is_file():
            raise ValueError(f'Starting Kitに必要なツールがありません: {name}')
    # On a newly built dataset, install the released fold map. Never replace a different map.
    save_once(work / 'folds.json', folds)
    link_directory(root / 'work', work)
    link_directory(root / 'starting_kit', kit)
    reference = json.loads((root / 'configs/reference_eval.json').read_text())
    save_once(root / 'results/eb_grid.json', reference)
    print('入力データとStarting Kitを接続しました。work/、starting_kit/、results/は公開対象外です。')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', required=True)
    ap.add_argument('--starting-kit', required=True)
    args = ap.parse_args()
    connect(args.work, args.starting_kit)


if __name__ == '__main__':
    main()
