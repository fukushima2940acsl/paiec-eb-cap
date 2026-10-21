"""Download the recorded public training snapshot from Hugging Face, CPU only.

Requires huggingface_hub and any access approval required by the upstream dataset.
No tokens are accepted on the command line or written to public files.
"""
import argparse
import json
from pathlib import Path
import shutil

from release_utils import ROOT


def download(output, root=ROOT):
    from huggingface_hub import HfApi, hf_hub_download
    record = json.loads((Path(root) / 'release/data_source.json').read_text())
    revision = record['revision']
    if not revision or not record['eligible_dirs']:
        raise ValueError('元データの固定版・ディレクトリ一覧がありません。保存済み学習データが必要です')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('出力先は空のディレクトリにしてください')
    repo = record['db_repo']
    available = set(HfApi().list_repo_files(repo_id=repo, repo_type='dataset', revision=revision))
    plan = []
    for directory in record['eligible_dirs']:
        if Path(directory).is_absolute() or '..' in Path(directory).parts or '\\' in directory:
            raise ValueError('データディレクトリが不正です')
        for name in ['benchmarks', 'subjects', 'items', 'response']:
            source = f'{directory}/{name}.parquet'
            if source not in available and name == 'response':
                source = f'{directory}/responses.parquet'
            if source not in available:
                raise ValueError(f'記録されたデータ版に必須テーブルがありません: {source}')
            plan.append((source, output / directory / f'{name}.parquet'))
    for source, destination in plan:
        cached = hf_hub_download(repo_id=repo, repo_type='dataset', revision=revision, filename=source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)
        print(f'取得: {source}', flush=True)
    (output / 'corpus_manifest.json').write_text(json.dumps(record, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, help='new or empty data directory')
    args = ap.parse_args()
    download(args.out)


if __name__ == '__main__':
    main()
