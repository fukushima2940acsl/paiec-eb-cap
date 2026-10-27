"""Inspect additional-data capacity first; optionally download the pinned public tables."""
import argparse
from pathlib import Path
import shutil
from common import HERE, ROOT, load, safe_relative, sha


def download(out, limit=None):
    record = load(HERE / 'release/data.json')
    entries = record['entries']
    total = sum(e['size'] for e in entries)
    print(f'Additional public tables: {len(entries)} files, {total / 2**30:.3f} GiB')
    print(f'Suggested free space including staging and scratch: {(3 * total + 4 * 2**30) / 2**30:.3f} GiB')
    if limit is None:
        print('Capacity inspection only. Nothing downloaded. Pass --max-download-gib to download.')
        return
    if not 0 < limit < float('inf') or total > limit * 2**30:
        raise ValueError('Download exceeds the explicit capacity limit')
    out = Path(out).resolve()
    if ROOT in out.parents and not any((ROOT / p) == out or (ROOT / p) in out.parents for p in ('payload', 'private')):
        raise ValueError('Use payload/, private/, or a separate external directory')
    if out == ROOT:
        raise ValueError('Do not download into the repository root')
    out.mkdir(parents=True, exist_ok=True)
    remaining = 0
    for entry in entries:
        target = out / safe_relative(entry['path'])
        if out not in target.resolve().parents:
            raise ValueError('Download destination escaped the chosen directory')
        if target.exists():
            if sha(target) != entry['sha256'] or target.stat().st_size != entry['size']:
                raise ValueError('Existing download differs; no overwrite: ' + entry['path'])
        else:
            remaining += entry['size']
    if shutil.disk_usage(out).free < 3 * remaining + 4 * 2**30:
        raise ValueError('Insufficient free disk space')
    from huggingface_hub import hf_hub_download
    for entry in entries:
        target = out / entry['path']
        if not target.exists():
            try:
                returned = Path(hf_hub_download(repo_id=record['repo'], repo_type='dataset',
                    revision=record['revision'], filename=entry['path'], local_dir=str(out))).resolve()
            except Exception as exc:
                raise RuntimeError('Download stopped: ' + type(exc).__name__ + '; authentication URLs are not printed') from None
            if returned != target.resolve():
                raise ValueError('Unexpected downloaded path')
        if target.stat().st_size != entry['size'] or sha(target) != entry['sha256']:
            raise ValueError('Downloaded size/hash mismatch')
        print('Verified:', entry['path'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=str(ROOT / 'payload/meanmatch-additional'))
    p.add_argument('--max-download-gib', type=float, help='Explicit bound; omit for capacity inspection only')
    args = p.parse_args()
    download(args.out, args.max_download_gib)
