"""Inspect allowed source files or Git's exact staged contents before publication."""
import argparse
from pathlib import Path
import re
import subprocess

from release_utils import ROOT

REQUIRED_RECORDS = {'configs/folds.json', 'configs/reference_eval.json', 'release/environment.json',
                    'release/data_fingerprints.json', 'release/data_source.json',
                    'release/starting_kit.json', 'release/submission_1008013.json'}
PRIVATE_ROOTS = {'work', 'payload', 'data', 'starting_kit', 'results', 'dist', 'private', '.git', '.venv', 'venv'}
PATTERNS = [r'gh[pousr]_[A-Za-z0-9]{20,}', r'github_pat_[A-Za-z0-9_]{20,}',
            r'hf_[A-Za-z0-9]{20,}', r'sk-[A-Za-z0-9_-]{20,}',
            r'-----BEGIN [A-Z ]*PRIVATE KEY-----']


def validate(entries, root=ROOT, require_records=True):
    root = Path(root)
    allowed = {line.strip() for line in (root / 'PUBLIC_FILES.txt').read_text().splitlines()
               if line.strip() and not line.startswith('#')}
    errors = []
    for name, mode, content in entries:
        if name not in allowed:
            errors.append(f'公開リスト外: {name}')
        if mode not in ['100644', '100755']:
            errors.append(f'通常ファイルではありません: {name}')
        if len(content) > 1024 * 1024 or b'\x00' in content:
            errors.append(f'大きなファイルまたはバイナリ: {name}')
        text = content.decode('utf-8', errors='replace')
        if any(re.search(pattern, text) for pattern in PATTERNS):
            errors.append(f'認証情報らしい文字列: {name}（内容は表示しません）')
    if require_records:
        missing = REQUIRED_RECORDS - {name for name, _, _ in entries}
        if missing:
            errors.append('サーバーでfinalize_repository.pyを実行し、記録を追加してください: ' + ', '.join(sorted(missing)))
    if errors:
        raise ValueError('\n'.join(errors))
    return len(entries)


def tree_entries(root=ROOT):
    root = Path(root)
    for p in sorted(root.rglob('*')):
        relative = p.relative_to(root)
        if relative.parts[0] in PRIVATE_ROOTS or '__pycache__' in relative.parts or p.name == '.DS_Store':
            continue
        if p.is_symlink():
            yield str(relative), '120000', b''
        elif p.is_file():
            yield str(relative), '100644', p.read_bytes()


def staged_entries(root=ROOT):
    root = Path(root).resolve()
    top = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=root, text=True).strip()
    if Path(top).resolve() != root:
        raise ValueError('公開用フォルダ自身がGitリポジトリではありません。git init -b mainを先に実行してください')
    raw = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=root)
    for record in raw.split(b'\0'):
        if not record:
            continue
        metadata, name = record.split(b'\t', 1)
        mode, oid, stage = metadata.decode().split()
        if stage != '0':
            raise ValueError('マージ競合が残っています')
        content = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=root)
        yield name.decode(), mode, content


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--staged', action='store_true', help='check the Git index instead of current source files')
    ap.add_argument('--template', action='store_true', help='allow not-yet-generated release records; local package QA only')
    args = ap.parse_args()
    entries = list(staged_entries() if args.staged else tree_entries())
    try:
        count = validate(entries, require_records=not args.template)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f'PASS: {count} files checked. データ・秘密情報の全種類を検出する保証はありません。git diff --cachedも確認してください。')


if __name__ == '__main__':
    main()
