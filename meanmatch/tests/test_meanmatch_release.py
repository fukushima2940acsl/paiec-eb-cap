"""Synthetic scientific checks and verification of the actual frozen artifact."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
from calibration import match_old_mean
from common import (Predictor, compare_prior, encoded, fit, fit_histograms, histograms,
                    load, safe_relative, save_new, sha, verify_baseline_prior)
from reproduce import check_inference, write_archive
from vendor.pair_stats import aggregate_benchmark, fit_prior
from validation import DataInvalid, connect_index, validate_benchmark


class MeanMatchReleaseTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame({'benchmark_id':['b','a','a','z','z'],
            'subject_key':['s1','s2','s3','s4','s5'], 'mean':[.8,.1,.4,.9,.7],
            'n_items':[80,1000,90,79,1001]})

    def test_histograms_match_frozen_estimator_and_cap(self):
        table = self.table()
        actual = fit_histograms(histograms(table))
        frozen = fit_prior(table, {b:0 for b in table.benchmark_id}, 'all', max_items=1000)
        self.assertEqual(encoded(actual), encoded(frozen))
        self.assertEqual(actual['training_pair_counts'], {'a':2,'b':1})
        self.assertEqual(encoded(actual), encoded(fit_histograms(histograms(table.iloc[::-1]))))

    def test_trials_are_averaged_within_item_first(self):
        frame = pd.DataFrame({'benchmark_id':['a']*4,'subject_key':['s']*4,
            'item_id':['x','x','x','y'],'response':[1,1,0,0]})
        result = aggregate_benchmark(frame,'a')
        self.assertAlmostEqual(result.iloc[0]['mean'], 1/3)
        self.assertEqual(result.iloc[0]['n_items'],2)

    def test_mean_target_uses_only_old_training(self):
        old = fit_histograms(histograms(self.table()))
        for value in (.2,.9):
            extra = pd.DataFrame({'benchmark_id':['new'],'subject_key':['s'],'mean':[value],'n_items':[100]})
            record = {'histograms':{'old':histograms(self.table()),'additional':histograms(extra)}}
            priors = fit(record)
            self.assertLess(abs(Predictor(priors['meanmatch']).posterior_mean(0,0)-Predictor(old).posterior_mean(0,0)),1e-12)
            self.assertEqual(priors['eb_cap'],old)
            self.assertFalse(priors['meanmatch']['calibration']['evaluation_labels_used'])

    def test_identity_tilt_and_mutation_rejected(self):
        prior = fit_histograms(histograms(self.table()))
        self.assertEqual(match_old_mean(prior,prior)['weights'],prior['weights'])
        bad = copy.deepcopy(prior); bad['training_pair_counts']['a'] += 1
        with self.assertRaises(ValueError): compare_prior(bad,prior)
        bad = copy.deepcopy(prior); bad['weights'][0] += .01
        with self.assertRaises(ValueError): compare_prior(bad,prior)

    def test_duplicate_and_invalid_histograms_rejected(self):
        rows = histograms(self.table())
        with self.assertRaises(ValueError): fit_histograms(rows+rows)
        bad = copy.deepcopy(rows);bad[0]['counts'][0]=-1
        with self.assertRaises(ValueError): fit_histograms(bad)
        table=self.table();table['n_items']=table['n_items'].astype(float);table.loc[0,'n_items']=80.5
        with self.assertRaises(ValueError): histograms(table)

    def test_no_conflicting_record_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'record.json'
            save_new(p,{'x':1});save_new(p,{'x':1})
            with self.assertRaises(ValueError): save_new(p,{'x':2})
            self.assertEqual(load(p),{'x':1})
        for name in ['../x','/absolute','a//b','a/./b']:
            with self.assertRaises(ValueError):safe_relative(name)

    def test_actual_submission_member_hashes_inference_and_zip(self):
        artifact=load(HERE/'release/artifact.json')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for name,source in [('model.py',HERE.parent/'submissions/eb_grid/model.py'),('prior.json',HERE/'artifact/prior.json')]:
                (out/name).write_bytes(source.read_bytes())
                self.assertEqual(sha(out/name),artifact['members_sha256'][name])
            self.assertEqual(check_inference(out,load(out/'prior.json'))['prediction_cases'],1056)
            archive=out/'submission.zip';write_archive(out,archive)
            self.assertEqual(sha(archive),artifact['zip_sha256'])
            with zipfile.ZipFile(archive) as z:self.assertEqual(sorted(z.namelist()),['model.py','prior.json'])

    def test_baseline_no_newline_serialization_regression(self):
        raw = (HERE/'artifact/eb_cap_prior.json').read_bytes()
        prior = json.loads(raw)
        expected = load(HERE.parent/'release/submission_1008013.json')['members_sha256']['prior.json']
        self.assertFalse(raw.endswith(b'\n'))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
        self.assertEqual(encoded(prior), raw + b'\n')
        # Reproduce the old false rejection without changing any model value.
        self.assertNotEqual(hashlib.sha256(encoded(prior)).hexdigest(), expected)
        result = verify_baseline_prior(prior)
        self.assertEqual(result['max_posterior_error'], 0)
        self.assertEqual(result['posterior_states'], 528)

    def test_baseline_roundoff_allowed_but_model_changes_rejected(self):
        prior = load(HERE/'artifact/eb_cap_prior.json')
        tiny = copy.deepcopy(prior)
        tiny['weights'][0] = float(np.nextafter(tiny['weights'][0], np.inf))
        tiny['mean'] = float(np.nextafter(tiny['mean'], np.inf))
        self.assertLessEqual(verify_baseline_prior(tiny)['max_posterior_error'], 1e-12)
        changed = copy.deepcopy(prior); changed['weights'][0] += 1e-5
        with self.assertRaises(ValueError): verify_baseline_prior(changed)
        changed = copy.deepcopy(prior); changed['training_pair_counts'][changed['training_benchmarks'][0]] += 1
        with self.assertRaises(ValueError): verify_baseline_prior(changed)
        with patch('common.sha', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'Canonical baseline prior bytes'):
                verify_baseline_prior(prior)

    def test_raw_validation_and_whole_benchmark_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths={n:root/n for n in ('benchmarks.parquet','subjects.parquet','items.parquet','response.parquet')}
            pd.DataFrame([{'benchmark_id':'new','release':'public','response_type':'binary','granularity':'item'}]).to_parquet(paths['benchmarks.parquet'])
            pd.DataFrame([{'subject_id':'s','normalized_name':'synthetic'}]).to_parquet(paths['subjects.parquet'])
            pd.DataFrame({'item_id':[str(i) for i in range(80)],'content':['synthetic '+str(i) for i in range(80)]}).to_parquet(paths['items.parquet'])
            responses=pd.DataFrame({'response_id':[str(i) for i in range(80)],'subject_id':['s']*80,'item_id':[str(i) for i in range(80)],'response':[i%2 for i in range(80)]})
            responses.to_parquet(paths['response.parquet'])
            conn=connect_index(root/'index.sqlite')
            try:
                table,report=validate_benchmark(conn,paths,'new',{}, {},root)
                self.assertEqual(report['n_cap_pairs'],1)
                self.assertEqual(table.iloc[0]['mean'],.5)
                responses.loc[0,'response']=2;responses.to_parquet(paths['response.parquet'])
                with self.assertRaises(DataInvalid): validate_benchmark(conn,paths,'new',{}, {},root)
            finally:conn.close()


if __name__=='__main__':unittest.main()
