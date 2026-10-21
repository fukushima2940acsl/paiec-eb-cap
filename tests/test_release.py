"""Release provenance and Git index tests; no network or actual data needed."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import check_public_source as audit
import download_recorded_data
import finalize_repository as finalizer
import release_utils
import verify_reproduction
import test_packaging as packaging


class ReleaseTests(unittest.TestCase):
    def setup_release(self, tmp):
        source, public = Path(tmp) / 'old', Path(tmp) / 'public'
        packaging.BuildTests().create_root(source)
        public.mkdir()
        for name in ['submissions/eb_grid/model.py', 'experiments/pair_stats.py', 'experiments/eb_core.py']:
            destination = public / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        with patch('pandas.read_parquet', side_effect=packaging.parquet_reader(packaging.fixture())):
            output = packaging.builder.build(source)
        return source, public, output / 'eb_cap_all.zip'

    def test_record_freeze_idempotence_and_model_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            source, public, archive = self.setup_release(tmp)
            originals = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
            with patch('importlib.metadata.version', return_value='1.2.3'):
                finalizer.finalize(source, root=public)
                before = {p: p.read_bytes() for p in public.rglob('*.json') if not p.is_symlink()}
                finalizer.finalize(source, root=public)
                self.assertEqual(before, {p: p.read_bytes() for p in before})
            for p, content in originals.items():
                self.assertEqual(p.read_bytes(), content)
            self.assertEqual((public / 'work').resolve(), (source / 'work').resolve())
            self.assertEqual((public / 'starting_kit').resolve(), (source / 'starting_kit').resolve())
            for p in (public / 'release').glob('*.json'):
                self.assertNotIn(str(tmp), p.read_text())
            self.assertIsNone(json.loads((public / 'release/data_source.json').read_text())['revision'])
            # Train again through the newly connected public repository, not by copying the ZIP.
            with patch('pandas.read_parquet', side_effect=packaging.parquet_reader(packaging.fixture())):
                rebuilt_directory = packaging.builder.build(public)
            verify_reproduction.verify(rebuilt_directory / 'eb_cap_all.zip', public)
            # A newly made ZIP with different date headers is still the same model.
            rebuilt = public / 'rebuilt.zip'
            contents = release_utils.archive_contents(archive)
            with ZipFile(rebuilt, 'w') as z:
                for name, content in contents.items():
                    z.writestr(ZipInfo(name, date_time=(2001, 1, 1, 0, 0, 0)), content)
            self.assertNotEqual(release_utils.sha_file(rebuilt), release_utils.sha_file(archive))
            verify_reproduction.verify(rebuilt, public)
            with ZipFile(rebuilt, 'w') as z:
                for name, content in contents.items():
                    z.writestr(name, content + (b' ' if name == 'prior.json' else b''))
            with self.assertRaisesRegex(ValueError, 'prior.json'):
                verify_reproduction.verify(rebuilt, public)

    def test_changed_dataset_and_ambiguous_zip_stop(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            source, public, archive = self.setup_release(tmp)
            (source / 'work/summary.csv').write_text('benchmark_id\nB\nA\n')
            with self.assertRaisesRegex(ValueError, '異なります'):
                finalizer.finalize(source, root=public)
            self.assertFalse((public / 'release').exists())
            extra = source / 'dist/eb_cap_all_other'
            extra.mkdir()
            (extra / 'eb_cap_all.zip').write_bytes(b'different')
            with self.assertRaisesRegex(ValueError, '複数'):
                release_utils.choose_archive(source)

    def test_manifest_export_is_allowlisted_and_paths_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            path = source / 'payload/data/corpus_manifest.json'
            path.parent.mkdir(parents=True)
            data = {'db_repo': 'aims-foundations/measurement-db', 'revision': 'a' * 40,
                    'eligible_dirs': ['example/formatted_tables'],
                    'files': ['example/formatted_tables/response.parquet'],
                    'token': 'a secret to omit', 'local_dir': str(tmp)}
            path.write_text(json.dumps(data))
            exported = finalizer.data_source_record(source)
            self.assertEqual(exported['revision'], 'a' * 40)
            self.assertNotIn('token', exported)
            self.assertNotIn(str(tmp), json.dumps(exported))
            data['eligible_dirs'] = ['../escape']
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                finalizer.data_source_record(source)

    def test_existing_different_record_or_link_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'record.json'
            release_utils.save_once(p, {'value': 1})
            with self.assertRaises(ValueError):
                release_utils.save_once(p, {'value': 2})
            self.assertEqual(json.loads(p.read_text()), {'value': 1})
            a, b, link = Path(tmp) / 'a', Path(tmp) / 'b', Path(tmp) / 'link'
            a.mkdir()
            b.mkdir()
            release_utils.link_directory(link, a)
            with self.assertRaises(ValueError):
                release_utils.link_directory(link, b)
            self.assertEqual(link.resolve(), a.resolve())


class DownloadTests(unittest.TestCase):
    def test_pinned_revision_and_response_filename(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            record = root / 'release/data_source.json'
            record.parent.mkdir()
            directory = 'sample/formatted_tables'
            revision = 'b' * 40
            record.write_text(json.dumps({'db_repo': 'aims-foundations/measurement-db',
                                         'revision': revision, 'eligible_dirs': [directory]}))
            cache = root / 'fake_cache'
            cache.write_bytes(b'synthetic parquet bytes')
            calls = []
            class API:
                def list_repo_files(self, **kwargs):
                    self.assert_revision(kwargs)
                    return [f'{directory}/{name}.parquet' for name in ['benchmarks', 'items', 'subjects', 'responses']]
                @staticmethod
                def assert_revision(kwargs):
                    assert kwargs['revision'] == revision
            def fetch(**kwargs):
                API.assert_revision(kwargs)
                calls.append(kwargs['filename'])
                return cache
            fake = types.SimpleNamespace(HfApi=API, hf_hub_download=fetch)
            with patch.dict(sys.modules, {'huggingface_hub': fake}):
                download_recorded_data.download(root / 'data', root=root)
            self.assertEqual(len(calls), 4)
            self.assertTrue((root / 'data' / directory / 'response.parquet').is_file())
            self.assertFalse((root / 'data' / directory / 'responses.parquet').exists())


class PublicAuditTests(unittest.TestCase):
    def test_reject_unknown_binary_link_and_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'PUBLIC_FILES.txt').write_text('ok.py\n')
            audit.validate([('ok.py', '100644', b'print(1)')], root, False)
            token = ('gh' + 'p_' + 'a' * 30).encode()
            for entry in [('data.parquet', '100644', b'data'), ('ok.py', '120000', b''),
                          ('ok.py', '100644', b'\x00'), ('ok.py', '100644', token)]:
                with self.assertRaises(ValueError):
                    audit.validate([entry], root, False)

    def test_reads_git_index_instead_of_unstaged_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(['git', 'init', '-q', '-b', 'main'], cwd=root, check=True)
            (root / 'PUBLIC_FILES.txt').write_text('ok.py\n')
            p = root / 'ok.py'
            p.write_text('gh' + 'p_' + 'a' * 30)
            subprocess.run(['git', 'add', 'ok.py'], cwd=root, check=True)
            p.write_text('safe working copy')
            with self.assertRaisesRegex(ValueError, '認証情報'):
                audit.validate(list(audit.staged_entries(root)), root, False)


if __name__ == '__main__':
    unittest.main()
