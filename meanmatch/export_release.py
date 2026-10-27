"""Capture sanitized public-training sufficient statistics from a verified original build."""
import argparse
import hashlib
from importlib import metadata
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile
import pandas as pd
from common import (HERE, check_baseline, compare_prior, fit, histograms,
                    load, safe_relative, save_new, sha, verify_baseline_prior)


def select_build(builder, explicit=None):
    expected = load(HERE / 'release/artifact.json')['zip_sha256']
    matches = []
    paths = [Path(explicit).resolve()] if explicit else sorted((builder / 'builds').glob('meanmatch_all_*'))
    for directory in paths:
        if not (directory / 'manifest.json').is_file():
            continue
        m = load(directory / 'manifest.json')
        if m.get('status') == 'checks_passed' and m.get('zip_sha256') == expected:
            matches.append(directory)
    if not matches:
        raise ValueError('No completed build matches the fixed submitted ZIP. Supply --build explicitly.')
    # Equivalent artifact bytes; never choose by hosted score or filesystem recency.
    return sorted(matches)[0]


def export(builder, build=None):
    check_baseline()
    builder = Path(builder).resolve(strict=True)
    for name, digest in load(HERE / 'release/original_builder_files.json').items():
        if sha(builder / safe_relative(name)) != digest:
            raise ValueError('Original builder source differs: ' + name)
    directory = select_build(builder, build)
    subprocess.run([sys.executable, '-B', str(builder / 'verify_submission.py'), '--build', str(directory)], check=True)
    manifest = load(directory / 'manifest.json')
    artifact = load(HERE / 'release/artifact.json')
    archive_path = directory / artifact['zip_name']
    if sha(archive_path) != artifact['zip_sha256']:
        raise ValueError('Selected build is not the fixed submitted artifact')
    with ZipFile(archive_path) as archive:
        for name, digest in artifact['members_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('Submitted member mismatch')
    old = pd.read_csv(directory / 'pair_stats.csv', float_precision='round_trip',
                      dtype={'benchmark_id': str, 'subject_key': str})
    new = pd.read_csv(directory / 'new_pair_stats.csv',
                      dtype={'benchmark_id': str, 'subject_key': str, 'excluded_folds': str})
    training = {'histograms': {'old': histograms(old), 'additional': histograms(new)},
                'source': 'Derived counts from verified public-training pair statistics; no private feedback.',
                'pair_stats_sha256': {n: sha(directory / n) for n in ('pair_stats.csv', 'new_pair_stats.csv')},
                'grid_size': 101, 'smoothing_per_bin': .001, 'min_items': 80, 'max_items': 1000,
                'old_target': 'Actual zero-label prediction fitted from all old public training data.'}
    priors = fit(training)
    if [(priors[m]['n_pairs'], priors[m]['n_benchmarks']) for m in ['eb_cap', 'expanded']] != [(5437, 30), (10187, 92)]:
        raise ValueError('Unexpected training population')
    compare_prior(priors['meanmatch'], load(HERE / 'artifact/prior.json'))
    baseline_check = verify_baseline_prior(priors['eb_cap'])
    print('PASS: baseline artifact hash, training composition and 528 posterior states; '
          'max prediction error = ' + str(baseline_check['max_posterior_error']), flush=True)
    info = manifest['input_provenance']['source_info']
    dataset = Path(info['dataset'])
    prepared = Path(info['validation_directory'])
    # Recheck recorded metadata; raw response data are not read or published here.
    for path, digest in info['metadata_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Original expansion metadata changed')
    plan = load(dataset / 'plan.json')
    downloads = load(dataset / 'download_manifest.json')
    cfg = load(HERE / 'candidates.json')
    if (plan['repo'] != cfg['repo'] or plan['revision'] != cfg['revision']
            or downloads['status'] != 'complete' or downloads['plan_sha256'] != sha(dataset / 'plan.json')):
        raise ValueError('Expansion download identity differs')
    entries = []
    for row in downloads['files']:
        relative = safe_relative(row['path'])
        if relative.rsplit('/', 1)[0] not in cfg['directories']:
            raise ValueError('Unexpected download directory')
        if not __import__('re').fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('Missing downloaded file hash')
        entries.append({'path': relative, 'size': row['size'], 'sha256': row['sha256']})
    if len(entries) != 396 or len({e['path'] for e in entries}) != 396:
        raise ValueError('Expected exactly 396 downloaded tables')
    validation = load(prepared / 'validation.json')
    reports = []
    for row in validation:
        reports.append({k: row[k] for k in ('slug', 'status', 'benchmark_id', 'reason', 'n_pairs', 'n_cap_pairs') if k in row})
    if len(reports) != 99 or sum(r['status'] == 'valid' for r in reports) != 96:
        raise ValueError('Unexpected quarantine inventory')
    public_data = {'repo': cfg['repo'], 'revision': cfg['revision'],
                   'entries': sorted(entries, key=lambda r: r['path']), 'validation': reports,
                   'raw_data_redistributed': False,
                   'export_scope': 'Metadata and cached-statistic reproduction; raw tables were not rehashed by this exporter.'}
    environment = {'original_build_versions': manifest['versions'],
                   'export_versions': {n: metadata.version(n) for n in ('numpy', 'pandas', 'pyarrow')},
                   'verification': 'Original saved build verification and histogram refit passed during export.',
                   'baseline_prior_verification': baseline_check,
                   'fresh_download_verified': False}
    check_baseline()
    save_new(HERE / 'release/training.json', training)
    save_new(HERE / 'release/data.json', public_data)
    save_new(HERE / 'release/environment.json', environment)
    print('PASS: exported public-training histogram counts, pinned data inventory and environment.')
    print('Baseline source and tag preserved. No raw data, local paths, credentials or private feedback exported.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--builder', required=True, help='Preserved paiec_eb_meanmatch_submit_20261025 directory')
    p.add_argument('--build', help='Optional exact completed build directory')
    args = p.parse_args()
    export(args.builder, args.build)
