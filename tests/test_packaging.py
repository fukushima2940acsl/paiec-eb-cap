import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

BASE = Path(__file__).resolve().parents[1]
PACKAGE = BASE / 'experiments'
sys.path.insert(0, str(PACKAGE))
from test_experiments import fixture, parquet_reader
from pair_stats import aggregate_benchmark, fit_prior
import pandas as pd

spec = importlib.util.spec_from_file_location('prepare_cap', BASE / 'prepare_eb_cap.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BuildTests(unittest.TestCase):
    def create_root(self, root, failing=False):
        for dirname in ['work', 'results', 'submissions/eb_grid', 'starting_kit/tools', 'experiments', 'dist']:
            (root / dirname).mkdir(parents=True, exist_ok=True)
        (root / 'work/folds.json').write_text(json.dumps({'seed': 0, 'n_folds': 2, 'folds': {'A': 0, 'B': 1}}))
        (root / 'work/summary.csv').write_text('benchmark_id\nA\nB\n')
        (root / 'work/items.parquet').write_bytes(b'items fixture')
        (root / 'work/subjects.parquet').write_bytes(b'subjects fixture')
        (root / 'work/responses.parquet').write_bytes(b'parquet IO is mocked in this test')
        (root / 'results/eb_grid.json').write_text(json.dumps({'args': {'draw_limit': 1000, 'n_draws': 2, 'seed': 0, 'n_folds': 2}}))
        (root / 'submissions/eb_grid/prior.json').write_text('preserve existing prior')
        (root / 'dist/existing.zip').write_bytes(b'preserve existing archive')
        shutil.copy2(PACKAGE / 'reference_model_fixture.txt', root / 'submissions/eb_grid/model.py')
        for name in ['pair_stats.py', 'eb_core.py']:
            shutil.copy2(PACKAGE / name, root / 'experiments' / name)
        # Local stubs test check orchestration; these are not official checkers.
        check = '''import sys
from zipfile import ZipFile
with ZipFile(sys.argv[1]) as z:
    assert sorted(z.namelist()) == ['model.py', 'prior.json']
print('stub ZIP check PASS')
'''
        if failing:
            check = "raise SystemExit(9)\n"
        (root / 'starting_kit/tools/check_submission_zip.py').write_text(check)
        (root / 'starting_kit/tools/run_smoke_test.py').write_text('''import sys, importlib.util, math
from pathlib import Path
p = Path(sys.argv[1]) / 'model.py'
s = importlib.util.spec_from_file_location('submission', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
for n in [0,1,3,7,15,31]:
    pair = [{'normalized_name':'same'}, {'benchmark_id':'x'}]
    pred = m.predict(pair, [[pair, i % 2] for i in range(n)])
    assert math.isfinite(pred) and 0 <= pred <= 1
print('stub smoke test PASS')
''')

    def test_all_folds_cap_packaging_and_original_preservation(self):
        data = fixture()
        with tempfile.TemporaryDirectory(prefix='cap test ') as tmp:
            root = Path(tmp)
            self.create_root(root)
            originals = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch('pandas.read_parquet', side_effect=parquet_reader(data)), contextlib.redirect_stdout(io.StringIO()):
                out = builder.build(root)
            for p, content in originals.items():
                self.assertEqual(p.read_bytes(), content)
            with ZipFile(out / 'eb_cap_all.zip') as z:
                self.assertEqual(sorted(z.namelist()), ['model.py', 'prior.json'])
                prior = json.loads(z.read('prior.json'))
                self.assertEqual(prior['n_pairs'], 4)
                self.assertEqual(prior['n_benchmarks'], 2)
                self.assertEqual(prior['max_items'], 1000)
                self.assertEqual(prior['fold'], 'all')
                self.assertEqual(prior['training_benchmarks'], ['A', 'B'])
                stats = pd.concat([aggregate_benchmark(data['responses.parquet'].query('benchmark_id == @bid'), bid)
                                   for bid in ['A', 'B']], ignore_index=True)
                self.assertEqual(prior, fit_prior(stats, {'A':0, 'B':1}, 'all', 1000))
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'checks_passed')
            self.assertEqual(manifest['zip_sha256'], builder.sha(out / 'eb_cap_all.zip'))
            self.assertEqual(manifest['inference_cases_checked'], 528)
            self.assertTrue((out / 'CHECKS_PASSED.txt').is_file())

    def test_failed_check_does_not_produce_submission_ready_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.create_root(root, failing=True)
            with patch('pandas.read_parquet', side_effect=parquet_reader(fixture())), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    builder.build(root)
            self.assertEqual(list((root / 'dist').rglob('eb_cap_all.zip')), [])
            self.assertEqual(list((root / 'dist').rglob('CHECKS_PASSED.txt')), [])

    def test_inventory_mismatch_stops_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.create_root(root)
            (root / 'work/summary.csv').write_text('benchmark_id\nA\n')
            with self.assertRaises(ValueError):
                builder.build(root)
            self.assertEqual(list((root / 'dist').glob('eb_cap_all_*')), [])


if __name__ == '__main__':
    unittest.main()
