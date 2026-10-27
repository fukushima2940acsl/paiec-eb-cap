"""Check the explicit public CV inventory, frozen sources and run records."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
METHODS = {'old': ['eb_cap', 'eb_cap_expanded', 'eb_cap_expanded_meanmatch'],
           'new': ['eb_old_only', 'eb_expanded', 'eb_expanded_meanmatch']}
COUNTS = {'old': (1528, 973, 30, 30, {3: 509, 4: 506, 5: 513}),
          'new': (2071, 1183, 60, 57, {3: 699, 4: 685, 5: 687})}
AGGREGATES = ['summary', 'by_seed', 'by_fold', 'by_budget', 'by_benchmark',
              'by_benchmark_budget', 'by_benchmark_seed', 'pair_balanced_summary',
              'benchmark_macro_summary', 'meanmatch_vs_expanded', 'meanmatch_vs_expanded_by_budget']
SECRET = re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|'
                    r'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|'
                    r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|'
                    r'https?://[^\s/"\x27]+:[^\s/"\x27]+@')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(parent, name):
    p = Path(name)
    if not p.parts or p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe relative path')
    parent = Path(parent).resolve()
    dest = parent / p
    current = parent
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink in public CV path')
    return dest


def check_files(root=HERE):
    expected = load(root / 'SOURCE_HASHES.json')
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != set(expected) | {'SOURCE_HASHES.json'}:
        raise ValueError('Public CV file inventory differs')
    for name, digest in expected.items():
        path = child(root, name)
        if sha(path) != digest:
            raise ValueError('Public CV file hash differs: ' + name)
        data = path.read_bytes()
        if len(data) > 1024 * 1024 or b'\x00' in data:
            raise ValueError('Unexpected binary/large public file: ' + name)
        text = data.decode('utf-8')
        if SECRET.search(text):
            raise ValueError('Possible credential in public CV file; contents hidden: ' + name)
        # Use split literals so these guards do not themselves contain forbidden paths.
        if any('/' + marker + '/' in text for marker in ('workspace', 'Users', 'home', 'root')):
            raise ValueError('Absolute server/user path in public CV file: ' + name)
        if re.search(r'\b(?:subject|benchmark)_\d{6,}\b', text):
            raise ValueError('Possible private hosted identifier in public CV file: ' + name)
    return expected


def check(root=HERE):
    files = check_files(root)
    baseline = load(root / 'BASELINE_FILES.json')
    for name, digest in baseline['files_sha256'].items():
        if sha(child(root.parent, name)) != digest:
            raise ValueError('Original baseline source differs: ' + name)
    if sha(root / 'source/calibration.py') != load(root / 'FROZEN_RULE.json')['sha256']:
        raise ValueError('Mean-matching rule differs')
    provenance = load(root / 'PROVENANCE.json')
    for name, record in provenance['exports'].items():
        if record['transformation'] == 'byte-identical copy' and files[name] != record['original_sha256']:
            raise ValueError('Byte-identical export claim failed: ' + name)
    for kind in METHODS:
        d = root / 'records' / kind
        r = load(d / 'run_record.json')
        if r['status'] != 'complete' or r['methods'] != METHODS[kind]:
            raise ValueError('Unexpected original run status/methods')
        if not all(r.get(k) is True for k in ('inputs_preserved', 'reference_reproduced',
                    'mean_constraint_passed', 'candidate_frozen_before_evaluation')):
            raise ValueError('Original run checks are incomplete')
        if sha(d / 'priors.json') != r['priors_sha256']:
            raise ValueError('Recorded prior identity differs')
        keys = {(f, s) for f in range(5) for s in (3, 4, 5)}
        snaps = r['snapshots']
        if len(snaps) != 15 or {(x['fold'], x['seed']) for x in snaps} != keys:
            raise ValueError('Snapshot inventory differs')
        prior_snaps = r['original_snapshot_inventory']
        normalize = lambda rows: {(x['fold'], x['seed']): (x['sha256'], x.get('pairs', x.get('n_pairs'))) for x in rows}
        if len(prior_snaps) != 15 or normalize(snaps) != normalize(prior_snaps):
            raise ValueError('Snapshot identity differs from preceding control run')
        controls = r['control_reproduction']
        if (len(controls) != 30 or {(x['fold'], x['seed'], x['method']) for x in controls} !=
                {(f, s, m) for f, s in keys for m in METHODS[kind][:2]} or
                any(not 0 <= x['max_absolute_score_error'] <= 1e-12 for x in controls)):
            raise ValueError('Original control reproduction evidence failed')
    print(f'PASS: {len(files)} public CV files, fixed calibration, original baseline, and completed-run records.')
    return files


if __name__ == '__main__':
    check()
