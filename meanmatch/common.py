"""Shared checks and exact sufficient-statistic reproduction for EB-Cap mean matching."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from calibration import match_old_mean
from vendor.eb_core import Predictor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def save_new(path, value):
    """An identical record is reusable; conflicting records are never overwritten."""
    path = Path(path)
    data = encoded(value)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Conflicting existing record: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_./-]+', name):
        raise ValueError('Invalid relative path')
    if name.startswith('/') or any(x in ('', '.', '..') for x in name.split('/')):
        raise ValueError('Unsafe relative path')
    return name


def check_baseline():
    """Read only: protect baseline code, release metadata, and the original Git tag."""
    for name, digest in load(HERE / 'release/baseline_files.json').items():
        if sha(ROOT / name) != digest:
            raise ValueError('Preserved baseline file differs: ' + name)
    artifact = load(HERE / 'release/artifact.json')
    if (ROOT / '.git').exists():
        for ref, expected in [('refs/tags/v0.1.0-eb-cap', artifact['baseline_tag_object']),
                              ('refs/tags/v0.1.0-eb-cap^{commit}', artifact['baseline_commit'])]:
            got = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', ref], text=True).strip()
            if got != expected:
                raise ValueError('Preserved baseline tag differs')


def validate_stats(table):
    required = ['benchmark_id', 'subject_key', 'mean', 'n_items']
    if table.empty or table[required].isna().any().any() or table.duplicated(required[:2]).any():
        raise ValueError('Missing or duplicate training pair statistics')
    if (not np.isfinite(table['mean']).all() or not table['mean'].between(0, 1).all()
            or not table.n_items.ge(1).all() or not table.n_items.eq(table.n_items.astype(int)).all()):
        raise ValueError('Invalid training rates or item counts')


def histograms(table):
    """Only derived public-training counts; no question text, subject IDs or test feedback."""
    validate_stats(table)
    eligible = table[table.n_items.between(80, 1000)]
    result = []
    for bid, group in eligible.groupby('benchmark_id'):
        idx = np.rint(group['mean'].to_numpy() * 100).astype(int)
        result.append({'benchmark_id': str(bid), 'counts': np.bincount(idx, minlength=101).tolist()})
    if not result:
        raise ValueError('No eligible pairs')
    return result


def fit_histograms(rows):
    """Same ordered arithmetic as frozen pair_stats.fit_prior after quantization."""
    if not rows or len({r['benchmark_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate benchmark histograms')
    grid = np.linspace(0, 1, 101)
    weights = np.zeros(101)
    counts = {}
    for row in sorted(rows, key=lambda r: r['benchmark_id']):
        hist = row['counts']
        if len(hist) != 101 or any(type(v) is not int or v < 0 for v in hist) or not sum(hist):
            raise ValueError('Invalid histogram counts')
        h = np.asarray(hist, dtype=float)
        weights += h / h.sum()
        counts[row['benchmark_id']] = sum(hist)
    weights += .001
    weights /= weights.sum()
    return {'grid': grid.round(6).tolist(), 'weights': weights.tolist(), 'fold': 'all',
            'n_benchmarks': len(counts), 'n_pairs': sum(counts.values()), 'max_items': 1000,
            'training_benchmarks': sorted(counts), 'training_pair_counts': counts,
            'mean': float(grid @ weights), 'tail0': float(weights[grid <= .02].sum()),
            'tail1': float(weights[grid >= .98].sum())}


def fit(record):
    old_rows, new_rows = record['histograms']['old'], record['histograms']['additional']
    if {r['benchmark_id'] for r in old_rows} & {r['benchmark_id'] for r in new_rows}:
        raise ValueError('Old/additional benchmark identity collision')
    old = fit_histograms(old_rows)
    expanded = fit_histograms(old_rows + new_rows)
    return {'eb_cap': old, 'expanded': expanded, 'meanmatch': match_old_mean(expanded, old)}


def compare_prior(actual, expected):
    """Allow numerical roundoff in training; still reject changed membership or rule."""
    if actual.keys() != expected.keys():
        raise ValueError('Prior fields differ')
    for name in actual:
        if name in ('grid', 'weights', 'mean', 'tail0', 'tail1'):
            if not np.allclose(actual[name], expected[name], atol=1e-14, rtol=0):
                raise ValueError('Prior numbers differ: ' + name)
        elif name == 'calibration':
            for key in expected[name]:
                a, b = actual[name][key], expected[name][key]
                if type(a) is float and type(b) is float:
                    if abs(a - b) > 1e-12:
                        raise ValueError('Calibration differs: ' + key)
                elif a != b:
                    raise ValueError('Calibration rule differs: ' + key)
        elif actual[name] != expected[name]:
            raise ValueError('Prior composition differs: ' + name)
    a, b = Predictor(actual), Predictor(expected)
    for n in range(32):
        for s in range(n + 1):
            if abs(a.posterior_mean(s, n) - b.posterior_mean(s, n)) > 1e-12:
                raise ValueError('Posterior prediction differs')


def verify_baseline_prior(actual):
    """Verify canonical bytes separately from the numerical training replay.

    The released baseline JSON has no trailing newline. New record serialization
    adds one, and floating-point reductions may also differ in the last bits.
    Neither changes the estimator. Keep the original artifact hash check, require
    exact training composition, and use the original builder's 1e-14 numerical
    tolerance plus a 1e-12 check of every posterior count state.
    """
    path = HERE / 'artifact/eb_cap_prior.json'
    expected_sha = load(ROOT / 'release/submission_1008013.json')['members_sha256']['prior.json']
    if sha(path) != expected_sha:
        raise ValueError('Canonical baseline prior bytes differ from submission 1008013')
    expected = load(path)
    compare_prior(actual, expected)
    a, b = Predictor(actual), Predictor(expected)
    max_prediction_error = max(abs(a.posterior_mean(s, n) - b.posterior_mean(s, n))
                               for n in range(32) for s in range(n + 1))
    return {'canonical_prior_sha256': expected_sha,
            'canonical_bytes_verified': True, 'training_composition_verified': True,
            'prior_numeric_atol': 1e-14, 'posterior_atol': 1e-12,
            'posterior_states': 528, 'max_posterior_error': max_prediction_error,
            'max_weight_error': float(np.max(np.abs(np.asarray(actual['weights']) - expected['weights'])))}
