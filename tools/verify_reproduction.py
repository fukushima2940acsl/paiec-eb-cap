"""Compare a rebuilt submission with the released member hashes (ZIP dates may differ)."""
import argparse
import json
from pathlib import Path

from release_utils import ROOT, archive_contents, choose_archive, member_hashes, sha_file


def verify(zip_path, root=ROOT):
    root = Path(root)
    record = json.loads((root / 'release/submission_1008013.json').read_text())
    actual = member_hashes(archive_contents(zip_path))
    if actual != record['members_sha256']:
        failed = [name for name in actual if actual[name] != record['members_sha256'].get(name)]
        raise ValueError('公開版と内容が一致しません: ' + ', '.join(failed))
    print('PASS: model.pyとprior.jsonが提出1008013の記録とバイト単位で一致しました。')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', type=Path, help='rebuilt ZIP; default: require one unique local candidate')
    ap.add_argument('--data', action='store_true', help='also verify full dataset SHA256 values (reads large files)')
    args = ap.parse_args()
    selected = choose_archive(ROOT, args.zip)
    verify(selected)
    if args.data:
        expected = json.loads((ROOT / 'release/data_fingerprints.json').read_text())['files']
        for name, info in expected.items():
            if sha_file(ROOT / 'work' / name) != info['sha256']:
                raise SystemExit(f'データのSHA256が一致しません: {name}')
            print(f'データ一致: {name}', flush=True)


if __name__ == '__main__':
    main()
