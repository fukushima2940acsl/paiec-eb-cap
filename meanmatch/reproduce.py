"""Refit the fixed prior, verify all count states, and package canonical submitted bytes."""
import argparse
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from common import (HERE, ROOT, Predictor, check_baseline, compare_prior, fit, load,
                    save_new, sha, verify_baseline_prior)


def check_inference(directory, prior):
    spec = importlib.util.spec_from_file_location('meanmatch_submission_check', directory / 'model.py')
    model = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(model)
    expected = Predictor(prior)
    from constants import SUBJECT_FIELDS
    subject = {k: '' for k in SUBJECT_FIELDS}
    subject['normalized_name'] = 'synthetic subject'
    item = {'benchmark_id': 'synthetic target', 'item_content': 'synthetic question',
            'item_features': '', 'interactors': ''}
    foreign = [[[{**subject, k: subject[k] + 'other'}, item], 1] for k in SUBJECT_FIELDS]
    foreign.append([[subject, {**item, 'benchmark_id': 'other'}], 1])
    for n in range(32):
        for s in range(n + 1):
            own = [[[subject, item], 1]] * s + [[[subject, item], 0]] * (n - s)
            for evidence in (own, foreign + own):
                value = model.predict([subject, item], evidence)
                if (type(value) is not float or not math.isfinite(value)
                        or abs(value - expected.posterior_mean(s, n)) > 1e-12):
                    raise ValueError('Prediction or subject/benchmark isolation mismatch')
    return {'count_states': 528, 'prediction_cases': 1056}


def write_archive(directory, path):
    with ZipFile(path, 'x', ZIP_DEFLATED) as archive:
        for name in ['model.py', 'prior.json']:
            info = ZipInfo(name, date_time=(2026, 10, 25, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (directory / name).read_bytes())


def reproduce(output, kit=None, artifact_only=False):
    check_baseline()
    record = load(HERE / 'release/artifact.json')
    if sha(HERE / 'calibration.py') != record['frozen_rule_sha256']:
        raise ValueError('Frozen mean-matching rule changed')
    sources = {'model.py': ROOT / 'submissions/eb_grid/model.py', 'prior.json': HERE / 'artifact/prior.json'}
    for name, path in sources.items():
        if sha(path) != record['members_sha256'][name]:
            raise ValueError('Canonical submitted member changed: ' + name)
    prior = load(sources['prior.json'])
    training = None
    baseline_check = None
    if not artifact_only:
        training = load(HERE / 'release/training.json')
        priors = fit(training)
        compare_prior(priors['meanmatch'], prior)
        baseline_check = verify_baseline_prior(priors['eb_cap'])
        target = Predictor(priors['eb_cap']).posterior_mean(0, 0)
        if abs(Predictor(prior).posterior_mean(0, 0) - target) > 1e-12:
            raise ValueError('Mean constraint failed')
    output = Path(output).resolve()
    # All outputs are kept outside publishable source paths.
    if ROOT in output.parents and not any((ROOT / n) == output or (ROOT / n) in output.parents for n in ('dist', 'private')):
        raise ValueError('Choose dist/, private/, or an output directory outside this repository')
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='meanmatch_', dir=output))
    submission = run / 'submission'; submission.mkdir()
    manifest = {'status': 'running', 'training_refit': not artifact_only, 'official_checks_passed': False,
                'source': 'Canonical submitted member bytes, after numerical refit verification unless artifact-only.',
                'members_sha256': record['members_sha256']}
    save_new(run / 'started.json', manifest)
    try:
        for name, path in sources.items():
            shutil.copyfile(path, submission / name)
        manifest['inference'] = check_inference(submission, prior)
        pending = run / 'candidate_not_for_submission.zip'
        write_archive(submission, pending)
        with ZipFile(pending) as archive:
            if sorted(archive.namelist()) != ['model.py', 'prior.json'] or archive.testzip() is not None:
                raise ValueError('Invalid archive')
            extracted = run / 'extracted'; extracted.mkdir()
            for name in archive.namelist():
                (extracted / name).write_bytes(archive.read(name))
        check_inference(extracted, prior)
        if kit is not None:
            kit = Path(kit).resolve(strict=True)
            checks = load(ROOT / 'release/starting_kit.json')['files']
            for name, target in [('check_submission_zip.py', pending), ('run_smoke_test.py', submission)]:
                script = kit / 'tools' / name
                if sha(script) != checks[name]:
                    raise ValueError('Official checker hash differs: ' + name)
                with (run / (name + '.log')).open('w') as log:
                    subprocess.run([sys.executable, '-B', str(script), str(target)], cwd=kit,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            manifest['official_checks_passed'] = True
        check_baseline()
        if any(sha(submission / n) != record['members_sha256'][n] for n in sources):
            raise ValueError('Submission members changed during checks')
        if sha(pending) != record['zip_sha256']:
            raise ValueError('ZIP bytes differ from submitted archive; inspect compression/runtime differences')
        final = run / record['zip_name']
        pending.rename(final)
        manifest.update(status='checks_passed', zip_sha256=sha(final), zip_name=final.name,
                        baseline_preserved=True)
        if training is not None:
            manifest['training_record_sha256'] = sha(HERE / 'release/training.json')
            manifest['baseline_prior_verification'] = baseline_check
        save_new(run / 'manifest.json', manifest)
        print('PASS: ' + str(final))
        print('Training refit:', not artifact_only, '| Official checks:', kit is not None)
        return run
    except BaseException as exc:
        save_new(run / 'failure.json', {'status': 'failed', 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=str(ROOT / 'dist'))
    p.add_argument('--official-kit', help='Matching preserved Codabench Starting Kit directory')
    p.add_argument('--artifact-only', action='store_true', help='Check packaged inference only; explicitly skip training refit')
    args = p.parse_args()
    reproduce(args.out, args.official_kit, args.artifact_only)
