"""Capture the server's exact release inputs and link local-only working folders.

No GitHub API calls, git commits, uploads, credentials, raw responses or private
Codabench feedback are included in the public records.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re

from connect_inputs import connect
from release_utils import ROOT, archive_contents, choose_archive, member_hashes, save_once, sha_file


def data_source_record(source):
    """Export only public repository metadata, not the whole local manifest."""
    path = Path(source) / 'payload/data/corpus_manifest.json'
    record = {'db_repo': 'aims-foundations/measurement-db', 'revision': None,
              'eligible_dirs': [], 'files': [],
              'note': 'Original download manifest unavailable; use the recorded prepared-data hashes.'}
    if not path.is_file():
        return record
    raw = json.loads(path.read_text())
    revision = raw.get('revision')
    if raw.get('db_repo') != record['db_repo'] or not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40,64}', revision):
        record['note'] = 'Original manifest has no recognized pinned revision; no revision was inferred.'
        return record
    for key in ['eligible_dirs', 'files']:
        values = raw.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) or not v
            or Path(v).is_absolute() or '..' in Path(v).parts or '\\' in v for v in values):
            raise ValueError('データ取得manifestのパス形式を確認してください')
        record[key] = sorted(set(values))
    record.update(revision=revision, note='Public training dataset revision recorded by the original download manifest.')
    return record


def signature(work):
    return {name: ((work / name).stat().st_size, (work / name).stat().st_mtime_ns)
            for name in ['responses.parquet', 'items.parquet', 'subjects.parquet', 'summary.csv', 'folds.json']}


def finalize(source, submission_zip=None, root=ROOT):
    root, source = Path(root).resolve(), Path(source).resolve(strict=True)
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('公開用フォルダは既存作業フォルダとは別の場所に置いてください（例: /workspace/paiec-eb-cap）')
    selected = choose_archive(source, submission_zip)
    contents = archive_contents(selected)
    members = member_hashes(contents)
    if contents['model.py'] != (root / 'submissions/eb_grid/model.py').read_bytes():
        raise ValueError('提出ZIPのmodel.pyと公開用コードが一致しません。変更せず確認してください')
    prior = json.loads(contents['prior.json'])
    if prior['fold'] != 'all' or prior.get('max_items') != 1000:
        raise ValueError('全データ学習のeb_cap提出物ではありません')
    if prior['n_pairs'] <= 0 or prior['n_benchmarks'] <= 0:
        raise ValueError('学習対象が空です')
    manifest_path = selected.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'checks_passed' or manifest['zip_sha256'] != sha_file(selected):
        raise ValueError('提出物の作成記録・ZIPハッシュを確認できません')
    if manifest['members_sha256'] != members:
        raise ValueError('提出物のメンバーハッシュが一致しません')
    work = source / 'work'
    start = signature(work)
    original = manifest['data_identity']
    if (original['responses_size'] != start['responses.parquet'][0]
            or original['responses_mtime_ns'] != start['responses.parquet'][1]
            or original['folds_sha256'] != sha_file(work / 'folds.json')
            or original['summary_sha256'] != sha_file(work / 'summary.csv')):
        raise ValueError('現在の学習データが提出ZIP作成時の記録と異なります')
    # Verify the recorded algorithm files against the actual submission build snapshot.
    for name in ['pair_stats.py', 'eb_core.py']:
        if sha_file(root / 'experiments' / name) != manifest['source_sha256'][name]:
            raise ValueError(f'提出時のコードと公開用コードが異なります: {name}')
    folds = json.loads((work / 'folds.json').read_text())
    old_args = json.loads((source / 'results/eb_grid.json').read_text())['args']
    keys = ['seed', 'n_folds', 'n_draws', 'draw_limit']
    args = {key: int(old_args[key]) for key in keys}
    if args['seed'] != folds['seed'] or args['n_folds'] != folds['n_folds'] or args['draw_limit'] != 1000:
        raise ValueError('既存の評価設定とfold割当が一致しません')
    args['fold'] = 'all'
    files = {}
    for name in start:
        print(f'再現用ハッシュを計算中: {name}', flush=True)
        files[name] = {'bytes': start[name][0], 'sha256': sha_file(work / name)}
    if signature(work) != start:
        raise ValueError('ハッシュ計算中にデータが変更されました')
    environment = {'python': platform.python_version(), 'packages': {
        name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'pyarrow']}}
    release = {'submission_id': 1008013, 'method': 'eb_cap', 'reported_status': 'Finished',
               'reported_brier_alc': .185216, 'reported_date_jst': '2026-10-21',
               'score_source': 'User-provided Codabench feedback; not an API verification by this script.',
               'artifact_binding': 'User associates the selected local ZIP with submission 1008013.',
               'zip_sha256': sha_file(selected), 'members_sha256': members,
               'training': {'fold': 'all', 'max_items': 1000, 'uniform_mix': 0., 'zero_half': False,
                            'n_pairs': prior['n_pairs'], 'n_benchmarks': prior['n_benchmarks'], 'mean': prior['mean']}}
    # Public records contain no absolute paths, user names, environment variables or tokens.
    save_once(root / 'configs/folds.json', folds)
    save_once(root / 'configs/reference_eval.json', {'args': args})
    save_once(root / 'release/environment.json', environment)
    save_once(root / 'release/data_fingerprints.json', {'files': files,
              'note': 'Hashes identify the exact prepared training files. Parquet bytes can differ after reserialization.'})
    save_once(root / 'release/data_source.json', data_source_record(source))
    save_once(root / 'release/starting_kit.json', {'files': {
        name: sha_file(source / 'starting_kit/tools' / name)
        for name in ['check_submission_zip.py', 'run_smoke_test.py']},
        'note': 'Hashes of the two local validation scripts; the complete Starting Kit is not redistributed.'})
    save_once(root / 'release/submission_1008013.json', release)
    connect(work, source / 'starting_kit', root=root)
    print(f'公開準備の記録を作成しました。提出1008013に対応づけたZIP: {selected}')
    print('GitHubへの送信は行っていません。次にtestsとcheck_public_source.pyを実行してください。')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, help='existing paiec_baseline directory')
    ap.add_argument('--submission-zip', help='ZIP actually uploaded as submission 1008013; otherwise require one unique candidate')
    args = ap.parse_args()
    finalize(args.source, args.submission_zip)


if __name__ == '__main__':
    main()
