"""Require all mean-match publication records and check frozen sources and aggregates."""
import re
import statistics
from common import (HERE, check_baseline, compare_prior, fit, load, safe_relative,
                    sha, verify_baseline_prior)


def check():
    check_baseline()
    for name, digest in load(HERE / 'release/source_hashes.json').items():
        if sha(HERE / safe_relative(name)) != digest:
            raise ValueError('Frozen addition file changed: ' + name)
    training = load(HERE / 'release/training.json')
    data = load(HERE / 'release/data.json')
    environment = load(HERE / 'release/environment.json')
    cfg = load(HERE / 'candidates.json')
    if data['repo'] != cfg['repo'] or data['revision'] != cfg['revision']:
        raise ValueError('Data snapshot differs')
    if len(data['entries']) != 396 or len({e['path'] for e in data['entries']}) != 396:
        raise ValueError('Incomplete raw-data inventory')
    for e in data['entries']:
        if safe_relative(e['path']).rsplit('/', 1)[0] not in cfg['directories']:
            raise ValueError('Unexpected data directory')
        if type(e['size']) is not int or e['size'] < 0 or not re.fullmatch('[0-9a-f]{64}', e['sha256']):
            raise ValueError('Invalid file identity')
    if not environment['original_build_versions'] or not environment['verification']:
        raise ValueError('Missing reproduction provenance')
    priors = fit(training)
    if [(priors[m]['n_pairs'], priors[m]['n_benchmarks']) for m in ('eb_cap','expanded')] != [(5437,30),(10187,92)]:
        raise ValueError('Unexpected training counts')
    compare_prior(priors['meanmatch'], load(HERE / 'artifact/prior.json'))
    verify_baseline_prior(priors['eb_cap'])
    results = load(HERE / 'release/reported_results.json')
    for method in ('eb_cap', 'meanmatch'):
        values = [r['score'] for r in results['runs'] if r['method'] == method and r['status'] == 'Finished']
        if len(values) != 3:
            raise ValueError('Repeat count differs')
        for key, actual in [('mean',statistics.mean(values)),('sample_sd',statistics.stdev(values)),('min',min(values)),('max',max(values))]:
            if abs(actual - results['summary'][method][key]) > 1e-12:
                raise ValueError('Reported repeat aggregate differs')
    # Public records should contain only explicitly selected scalar metadata.
    for name in ('training.json','data.json','environment.json','reported_results.json'):
        text = (HERE / 'release' / name).read_text()
        if any(t in text for t in ('/workspace/', '/Users/')) or re.search(r'\b(?:subject|benchmark)_\d{6,}\b', text):
            raise ValueError('Local path or private feedback identifier in public record: ' + name)
    print('PASS: complete mean-match records, frozen code, fitted prior and aggregate results.')
    print('No publication, submission or Git modification performed.')


if __name__ == '__main__':
    check()
