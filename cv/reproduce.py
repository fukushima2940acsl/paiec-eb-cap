"""Refit CV priors from public training statistics and replay paired result tables."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from check_release import HERE, METHODS, COUNTS, AGGREGATES, load, sha, check


def save(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def equal_prior(a, b, location='prior'):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError('Prior keys differ: ' + location)
        for k in a:
            equal_prior(a[k], b[k], location + '/' + k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError('Prior list lengths differ: ' + location)
        for x, y in zip(a, b):
            equal_prior(x, y, location)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > 1e-12:
            raise ValueError('Prior numeric discrepancy: ' + location)
    elif a != b:
        raise ValueError('Prior metadata differs: ' + location)


def compare_frames(a, b, name):
    import numpy as np
    import pandas as pd
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        raise ValueError('Aggregate columns/shape differ: ' + name)
    maximum = 0.0
    for c in a:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if not np.allclose(a[c], b[c], atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError('Aggregate numeric discrepancy: ' + name + '/' + c)
            diff = (a[c] - b[c]).abs().dropna()
            maximum = max(maximum, float(diff.max()) if len(diff) else 0.0)
        elif not a[c].equals(b[c]):
            raise ValueError('Aggregate row labels differ: ' + name + '/' + c)
    return maximum


def fit_population(kind):
    import pandas as pd
    sys.path.insert(0, str(HERE / 'source'))
    from calibration import match_old_mean, refit_controls
    from training_old import fit_priors
    from vendor.eb_core import Predictor
    d = HERE / 'records' / kind
    # Retain different historical parsing rules; no reserialization of training CSVs.
    options = {'dtype': {'benchmark_id': str, 'subject_key': str}}
    old = pd.read_csv(d / 'pair_stats.csv', **options,
                      **({'float_precision': 'round_trip'} if kind == 'old' else {}))
    new = pd.read_csv(d / 'new_pair_stats.csv', dtype={'benchmark_id': str, 'subject_key': str, 'excluded_folds': str})
    split = load(d / ('folds.json' if kind == 'old' else 'split.json'))
    recorded = load(d / 'priors.json')
    fitted, maximum_b0, maximum_prediction = {}, 0.0, 0.0
    methods = METHODS[kind]
    for fold in range(5):
        priors = fit_priors(old, new, split['folds'], fold) if kind == 'old' else refit_controls(old, new, split, fold)
        priors[methods[2]] = match_old_mean(priors[methods[1]], priors[methods[0]])
        equal_prior(priors, recorded[str(fold)])
        for method in methods:
            a, b = Predictor(priors[method]), Predictor(recorded[str(fold)][method])
            for n in range(32):
                for s in range(n + 1):
                    maximum_prediction = max(maximum_prediction, abs(a.posterior_mean(s, n) - b.posterior_mean(s, n)))
        gap = abs(Predictor(priors[methods[0]]).posterior_mean(0, 0) - Predictor(priors[methods[2]]).posterior_mean(0, 0))
        maximum_b0 = max(maximum_b0, gap)
        fitted[fold] = priors
    if maximum_b0 > 1e-12 or maximum_prediction > 1e-12:
        raise ValueError('Mean constraint or posterior parity failed')
    return fitted, split, maximum_b0, maximum_prediction


def worker(kind, output):
    import numpy as np
    import pandas as pd
    check()
    # Prior fitting finishes before any evaluation score table is opened.
    fitted, split, max_b0, max_prediction = fit_population(kind)
    save(output / 'refitted_priors.json', fitted)
    comparison = importlib.import_module('comparison_' + kind)
    d = HERE / 'records' / kind
    methods = METHODS[kind]
    options = {'dtype': {'benchmark': str, 'subject_key': str}}
    tables = {m: pd.read_csv(d / 'pairs' / (m + '.csv'), **options) for m in methods}
    aligned = comparison.align(tables)
    base = tables[methods[0]]
    groups = {b: b for b in split['folds']} if kind == 'old' else split['benchmark_to_group']
    actual = (len(base), len(base.drop_duplicates(['benchmark', 'subject_key'])), base.benchmark.nunique(),
              len({groups[b] for b in base.benchmark}), base.groupby('seed').size().to_dict())
    if actual != COUNTS[kind]:
        raise ValueError('Evaluation population differs from archived run')
    for row in base[['fold', 'benchmark']].drop_duplicates().itertuples(index=False):
        held = ([b for b, f in split['folds'].items() if f == row.fold] if kind == 'old'
                else split['folds'][row.fold]['evaluation_benchmarks'])
        if row.benchmark not in held:
            raise ValueError('Scored benchmark is not held out in its recorded fold')
    if np.max(np.abs(aligned[methods[0]].B0 - aligned[methods[2]].B0)) > 1e-12:
        raise ValueError('Paired zero-label scores differ')
    for snap in load(d / 'run_record.json')['snapshots']:
        if len(base[(base.fold == snap['fold']) & (base.seed == snap['seed'])]) != snap['pairs']:
            raise ValueError('Snapshot row count differs')
    for method in methods[:2]:
        previous = pd.read_csv(d / 'reference_controls' / (method + '.csv'), **options)
        a, b = aligned[method], previous.set_index(comparison.KEYS).sort_index()
        if not a.index.equals(b.index):
            raise ValueError('Reference control pair identity differs')
        for col in comparison.BCOLS + ['ALC', 'n_acquired', 'n_eval_items']:
            if not np.allclose(a[col], b[col], atol=1e-12, rtol=0):
                raise ValueError('Reference control score/count differs')
    comparison.compare(tables, output, groups)
    comparison.calibration_contrast(tables, output, groups)
    errors = {name: compare_frames(pd.read_csv(d / 'expected' / (name + '.csv')),
                                   pd.read_csv(output / (name + '.csv')), name) for name in AGGREGATES}
    reported = load(HERE / 'reported_summary.json')['populations']['old' if kind == 'old' else 'additional']
    scores = pd.read_csv(output / 'summary.csv').set_index('method')
    budgets = pd.read_csv(output / 'by_budget.csv').set_index('method')
    for method, label in zip(methods, ('old_only', 'expanded', 'meanmatch')):
        for column, value in reported['summary'][label].items():
            if abs(scores.loc[method, column] - value) > 5.1e-7:
                raise ValueError('Rounded report summary differs')
        for budget, value in zip(comparison.BCOLS, reported['budgets'][label]):
            if abs(budgets.loc[method, budget] - value) > 5.1e-7:
                raise ValueError('Rounded report budget table differs')
    check()
    checks = {'status': 'passed', 'population': kind, 'rows': actual[0], 'distinct_pairs': actual[1],
              'benchmarks': actual[2], 'clusters': actual[3], 'seed_rows': actual[4],
              'prior_refit_before_score_table_read': True, 'max_B0_mean_error': max_b0,
              'max_posterior_difference': max_prediction, 'aggregate_max_abs_errors': errors,
              'reported_rounded_values_match': True, 'full_raw_data_scoring_performed': False,
              'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__}}
    save(output / 'CHECKS.json', checks)
    print('PASS:', kind, 'prior refit, posterior parity, controls, eleven aggregate tables and report values.')


def output_parent(value):
    if (HERE.parent / 'results').is_symlink():
        raise ValueError('The results directory must not be a symlink')
    p = Path(value).resolve()
    allowed = (HERE.parent / 'results').resolve()
    if p != allowed and allowed not in p.parents:
        raise ValueError('Choose an output directory under the ignored repository results/ directory')
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(HERE.parent / 'results'), help='Parent under ignored results/; creates a unique run')
    ap.add_argument('--worker', choices=['old', 'new'], help=argparse.SUPPRESS)
    ap.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        dest = output_parent(args.worker_output)
        if any(dest.iterdir()):
            raise ValueError('Worker output is not empty')
        worker(args.worker, dest)
        return
    original = check()
    out = Path(tempfile.mkdtemp(prefix='cv_replay_', dir=output_parent(args.out)))
    manifest = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'public_files_sha256': original, 'source_archive_sha256': load(HERE / 'PROVENANCE.json')['source_archive_sha256'],
                'scope': 'Saved training-statistic and score-table replay; no raw-data evaluation or new benchmark sampling.'}
    save(out / 'manifest.json', manifest)
    try:
        for kind in ('old', 'new'):
            dest = out / kind
            dest.mkdir()
            subprocess.run([sys.executable, '-B', str(HERE / 'reproduce.py'), '--worker', kind,
                            '--worker-output', str(dest)], check=True)
        if check() != original:
            raise ValueError('Public inputs changed during replay')
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                        public_inputs_preserved=True, checks={k: load(out / k / 'CHECKS.json') for k in ('old', 'new')})
        print('PASS: both public CV comparisons reproduced. Results:', out)
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
