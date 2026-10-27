"""Tests for release checks, scientific isolation, and replay failure conditions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

CV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CV))
sys.path.insert(0, str(CV / 'source'))
import numpy as np
import pandas as pd
import check_release as checks
import reproduce
from training_old import fit_priors
from calibration import match_old_mean, refit_controls
from comparison_old import align
from vendor.eb_core import Predictor


def stats(benchmarks, means):
    return pd.DataFrame({'benchmark_id': benchmarks, 'subject_key': 'public-key', 'mean': means, 'n_items': 80})


class ReleaseTests(unittest.TestCase):
    def test_publication_inventory_and_baseline(self):
        checks.check()

    def test_tampered_and_extra_files_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = root / 'x.json'; p.write_text('{}')
            (root / 'SOURCE_HASHES.json').write_text(json.dumps({'x.json': checks.sha(p)}))
            checks.check_files(root)
            p.write_text('{"changed": true}')
            with self.assertRaises(ValueError): checks.check_files(root)
            p.write_text('{}')
            (root / 'extra.json').write_text('{}')
            with self.assertRaises(ValueError): checks.check_files(root)

    def test_absolute_path_and_secret_not_allowed(self):
        for text in ('/' + 'workspace' + '/private', 'github_pat_' + 'X' * 30):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                p = root / 'x.json'; p.write_text(json.dumps({'value': text}))
                (root / 'SOURCE_HASHES.json').write_text(json.dumps({'x.json': checks.sha(p)}))
                with self.assertRaises(ValueError): checks.check_files(root)

    def test_prior_and_result_discrepancies_rejected(self):
        with self.assertRaises(ValueError): reproduce.equal_prior({'weights': [0.5, 0.5]}, {'weights': [0.4, 0.6]})
        good = pd.DataFrame({'method': ['a'], 'ALC': [.2]})
        self.assertEqual(reproduce.compare_frames(good, good, 'fixture'), 0)
        for bad in (pd.DataFrame({'method': ['a'], 'ALC': [.3]}), pd.DataFrame({'method': ['b'], 'ALC': [.2]})):
            with self.assertRaises(ValueError): reproduce.compare_frames(good, bad, 'fixture')

    def test_outputs_cannot_replace_public_records(self):
        with self.assertRaises(ValueError): reproduce.output_parent(CV / 'records')

    def test_old_held_labels_cannot_change_training_target(self):
        old = stats(['held', 'train'], [.2, .6])
        new = stats(['extra', 'related'], [.8, .1]); new['excluded_folds'] = ['[]', '[0]']
        first = fit_priors(old, new, {'held': 0, 'train': 1}, 0)
        old.loc[old.benchmark_id == 'held', 'mean'] = .99
        new.loc[new.benchmark_id == 'related', 'mean'] = .99
        second = fit_priors(old, new, {'held': 0, 'train': 1}, 0)
        self.assertEqual(first, second)
        calibrated = match_old_mean(second['eb_cap_expanded'], second['eb_cap'])
        self.assertLess(abs(Predictor(calibrated).posterior_mean(0, 0) - Predictor(second['eb_cap']).posterior_mean(0, 0)), 1e-12)

    def test_new_overlap_group_is_excluded_from_all_training_arms(self):
        old = stats(['old', 'related_old'], [.4, .2])
        new = stats(['held', 'extra'], [.8, .6])
        split = {'folds': [{'fold': 0, 'excluded_old_benchmarks': ['related_old'],
                           'excluded_new_benchmarks': ['held'], 'evaluation_benchmarks': ['held'], 'evaluation_groups': ['g']}],
                 'groups': [{'group': 'g', 'members': ['held', 'related_old']}]}
        a = refit_controls(old, new, split, 0)
        old.loc[old.benchmark_id == 'related_old', 'mean'] = .99
        new.loc[new.benchmark_id == 'held', 'mean'] = .01
        self.assertEqual(a, refit_controls(old, new, split, 0))
        bad = deepcopy(split); bad['folds'][0]['excluded_old_benchmarks'] = []
        with self.assertRaises(ValueError): refit_controls(old, new, bad, 0)

    def test_paired_identity_and_alc_are_required(self):
        row = {'seed': 3, 'fold': 0, 'benchmark': 'a', 'subject_key': 's', 'n_eval_items': 40, 'n_acquired': 31,
               **{f'B{n}': .2 for n in [0, 1, 3, 7, 15, 31]}, 'ALC': .2}
        good = pd.DataFrame([row]); align({'eb_cap': good, 'candidate': good})
        for bad in (pd.DataFrame([{**row, 'subject_key': 'other'}]), pd.DataFrame([{**row, 'ALC': .3}]), pd.concat([good, good])):
            with self.assertRaises(ValueError): align({'eb_cap': good, 'candidate': bad})


if __name__ == '__main__':
    unittest.main()
