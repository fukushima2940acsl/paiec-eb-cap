"""Recompute the released histograms from old prepared tables and additional raw data."""
import argparse
import io
from pathlib import Path
import tempfile
import pandas as pd
import pyarrow as pa
from common import HERE, ROOT, check_baseline, histograms, load, safe_relative, save_new, sha
from validation import DataInvalid, connect_index, index_old_items, validate_benchmark
from vendor.pair_stats import aggregate_benchmark

pa.set_cpu_count(4)
pa.set_io_thread_count(2)


def prepare(old_prepared, new_data, output):
    check_baseline()
    old_prepared, new_data, output = [Path(p).resolve() for p in (old_prepared, new_data, output)]
    if output == ROOT or (ROOT in output.parents and not any((ROOT / p) == output or (ROOT / p) in output.parents for p in ('private', 'dist'))):
        raise ValueError('Choose private/, dist/, or an external output directory')
    if any(output == p or p in output.parents for p in (old_prepared, new_data)):
        raise ValueError('Output must be separate from training inputs')
    cfg = load(HERE / 'candidates.json')
    public = load(HERE / 'release/data.json')
    frozen = load(HERE / 'release/training.json')['histograms']
    fold_record = load(ROOT / 'configs/folds.json')
    if load(old_prepared / 'folds.json') != fold_record:
        raise ValueError('Old prepared fold identities differ from the baseline')
    # Different Parquet encodings may represent the same data; final histograms
    # are the exact semantic check. Record the consumed files without rewriting.
    before = {str(old_prepared / n): sha(old_prepared / n) for n in ('responses.parquet', 'items.parquet', 'folds.json')}
    grouped = {}
    for entry in public['entries']:
        path = new_data / safe_relative(entry['path'])
        if new_data not in path.resolve().parents:
            raise ValueError('Input path escaped the additional-data directory')
        if path.stat().st_size != entry['size'] or sha(path) != entry['sha256']:
            raise ValueError('Additional raw input differs: ' + entry['path'])
        grouped.setdefault(entry['path'].rsplit('/', 1)[0], {})[path.name] = path
        before[str(path)] = entry['sha256']
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='meanmatch_data_', dir=output))
    folds = fold_record['folds']
    pieces = []
    for bid in sorted(folds):
        responses = pd.read_parquet(old_prepared / 'responses.parquet',
            columns=['benchmark_id', 'subject_key', 'item_id', 'response'], filters=[('benchmark_id', 'in', [bid])])
        pieces.append(aggregate_benchmark(responses, bid))
        print('Old training statistics:', bid, flush=True)
    old = pd.concat(pieces, ignore_index=True)
    old.to_csv(run / 'pair_stats.csv', index=False)
    old = pd.read_csv(run / 'pair_stats.csv', float_precision='round_trip', dtype={'benchmark_id': str, 'subject_key': str})
    conn = connect_index(run / 'scratch.sqlite')
    reports, pieces = [], []
    try:
        index_old_items(conn, old_prepared / 'items.parquet', folds, run)
        for directory in cfg['directories']:
            slug = directory.split('/')[0]
            try:
                stats, report = validate_benchmark(conn, grouped[directory], slug, folds, cfg['family_exclusions'], run)
                pieces.append(stats)
            except DataInvalid as exc:
                conn.rollback()
                report = {'slug': slug, 'status': 'quarantined', 'reason': str(exc)}
            reports.append({k: report[k] for k in ('slug', 'status', 'benchmark_id', 'reason', 'n_pairs', 'n_cap_pairs') if k in report})
            print(slug + ': ' + report['status'], flush=True)
        if reports != public['validation']:
            raise ValueError('Validation/quarantine outcome differs from the recorded release')
        new = pd.concat(pieces, ignore_index=True)
        new.to_csv(run / 'new_pair_stats.csv', index=False)
        # Preserve the original builder's asymmetric CSV parsing precisely.
        new = pd.read_csv(run / 'new_pair_stats.csv', dtype={'benchmark_id': str, 'subject_key': str})
        actual = {'old': histograms(old), 'additional': histograms(new)}
        if actual != frozen:
            raise ValueError('Raw-derived training histograms differ; do not retune or overwrite release records')
        for path, digest in before.items():
            if sha(path) != digest:
                raise ValueError('Training input changed during preparation')
        check_baseline()
        save_new(run / 'histograms.json', {'histograms': actual})
        save_new(run / 'manifest.json', {'status': 'checks_passed', 'input_files_sha256': before,
                    'histograms_match_release': True, 'baseline_preserved': True})
        print('PASS: raw public tables reproduce the released histogram counts:', run)
    finally:
        conn.close()
        for suffix in ('', '-journal', '-wal', '-shm'):
            (run / ('scratch.sqlite' + suffix)).unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-prepared', default=str(ROOT / 'work'))
    p.add_argument('--new-data', default=str(ROOT / 'payload/meanmatch-additional'))
    p.add_argument('--out', default=str(ROOT / 'private'))
    a = p.parse_args()
    prepare(a.old_prepared, a.new_data, a.out)
