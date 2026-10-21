"""Synthetic correctness tests; parquet IO is mocked, not a performance evaluation."""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from eb_core import Predictor
import local_eval_v2 as evaluator
from pair_stats import aggregate_benchmark, fit_prior
import run_experiments as runner

PACKAGE = Path(__file__).resolve().parent


def fixture():
    responses, items, subjects = [], {}, []
    for bid, skey, count, mode in [("A", "a", 80, 0), ("A", "b", 81, 1),
                                    ("A", "c", 1001, 2), ("B", "d", 83, 3), ("B", "e", 80, 4)]:
        subject = {key: "" for key in evaluator.SUBJECT_FIELDS}
        subject.update(subject_key=skey, normalized_name="same displayed name", harness=skey)
        subjects.append(subject)
        for i in range(count):
            iid = f"item_{i:04}"
            items[bid, iid] = {"benchmark_id": bid, "item_id": iid, "item_content": iid, "item_features": ""}
            ys = [int((i + mode) % (mode + 2) == 0)]
            if i % 9 == 0:
                ys.append(1 - ys[0])
            for y in ys:
                responses.append({"benchmark_id": bid, "subject_key": skey, "item_id": iid,
                                  "response": y, "interactors": ""})
    return {"responses.parquet": pd.DataFrame(responses), "items.parquet": pd.DataFrame(items.values()),
            "subjects.parquet": pd.DataFrame(subjects)}


def parquet_reader(data):
    def read(path, columns=None, filters=None, **kwargs):
        table = data[Path(path).name].copy()
        for name, operation, value in filters or []:
            if operation != "in":
                raise AssertionError("unexpected filter")
            table = table[table[name].isin(value)]
        return table if columns is None else table[columns]
    return read


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.prior = {"grid": [0, .3, .8, 1], "weights": [.2, .4, .1, .3]}

    def test_legacy_parity_all_528_count_combinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "model.py").write_text((PACKAGE / "reference_model_fixture.txt").read_text())
            (path / "prior.json").write_text(json.dumps(self.prior))
            _, legacy, _ = evaluator.load_submission(str(path))
            current = Predictor(self.prior)
            for n in range(32):
                for s in range(n + 1):
                    self.assertAlmostEqual(current.posterior_mean(s, n), legacy.posterior_mean(s, n), places=14)

    def test_mixture_updates_component_weight_by_evidence(self):
        # G = point mass at 1/2; two successes give L_G=1/4, L_uniform=1/3.
        current = Predictor({"grid": [.5], "weights": [1.]}, uniform_mix=.25)
        expected = ((.75 / 4) * .5 + (.25 / 3) * .75) / (.75 / 4 + .25 / 3)
        self.assertAlmostEqual(current.posterior_mean(2, 2), expected, places=14)
        self.assertNotAlmostEqual(expected, .75 * .5 + .25 * .75, places=6)
        uniform = Predictor(self.prior, uniform_mix=1)
        for n in range(32):
            for s in range(n + 1):
                self.assertEqual(uniform.posterior_mean(s, n), (s + 1) / (n + 2))

    def test_zero_half_only_changes_zero_matching_labels(self):
        original = Predictor(self.prior)
        candidate = Predictor(self.prior, zero_half=True)
        self.assertEqual(candidate.posterior_mean(0, 0), .5)
        for n in [1, 3, 7, 15, 31]:
            for s in range(n + 1):
                self.assertEqual(original.posterior_mean(s, n), candidate.posterior_mean(s, n))
        subject = {"normalized_name": "same", "harness": "one"}
        other = {"normalized_name": "same", "harness": "two"}
        item = {"benchmark_id": "x"}
        foreign = [[[other, item], 1], [[subject, {"benchmark_id": "y"}], 1]]
        self.assertEqual(candidate.predict([subject, item], foreign), .5)
        self.assertEqual(candidate.predict([subject, item], foreign + [[[subject, item], 0]]),
                         candidate.posterior_mean(0, 1))

    def test_extreme_prior_is_finite_for_all_budgets(self):
        for weights in ([1e-300, 1.], [1., 1e-300]):
            for mix in [0, .25, 1]:
                model = Predictor({"grid": [0., 1.], "weights": weights}, uniform_mix=mix)
                for n in [0, 1, 3, 7, 15, 31]:
                    for s in range(n + 1):
                        p = model.posterior_mean(s, n)
                        self.assertTrue(math.isfinite(p) and 0 <= p <= 1)


class TrainingTests(unittest.TestCase):
    def test_item_weighting_is_distinct_from_response_weighting(self):
        table = pd.DataFrame([("A", "s", "i", 0), ("A", "s", "i", 0),
                              ("A", "s", "i", 1), ("A", "s", "j", 1)],
                             columns=["benchmark_id", "subject_key", "item_id", "response"])
        stats = aggregate_benchmark(table, "A")
        self.assertAlmostEqual(stats.iloc[0]["mean"], 2 / 3)
        self.assertEqual(stats.iloc[0]["n_items"], 2)

    def test_held_out_labels_do_not_change_prior_and_cap_is_inclusive(self):
        stats = pd.DataFrame([("A", "one", .3, 80), ("A", "two", .6, 1000),
                              ("A", "three", .9, 1001), ("A", "four", 0., 79),
                              ("B", "holdout", .1, 100)],
                             columns=["benchmark_id", "subject_key", "mean", "n_items"])
        folds = {"A": 0, "B": 1}
        before = fit_prior(stats, folds, 1, 1000)
        stats.loc[stats.benchmark_id == "B", "mean"] = 1.
        after = fit_prior(stats, folds, 1, 1000)
        self.assertEqual(before, after)
        self.assertEqual(before["n_pairs"], 2)
        self.assertEqual(before["training_benchmarks"], ["A"])
        self.assertEqual(fit_prior(stats, folds, 1)["n_pairs"], 3)

    def test_aggregated_prior_matches_original_trainer_all_folds(self):
        data = fixture()
        stats = pd.concat([aggregate_benchmark(data["responses.parquet"].query("benchmark_id == @bid"), bid)
                           for bid in ["A", "B"]], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp, patch("pandas.read_parquet", side_effect=parquet_reader(data)):
            root = Path(tmp)
            (root / "folds.json").write_text(json.dumps({"folds": {"A": 0, "B": 1}}))
            for fold in [0, 1, "all"]:
                argv = [str(PACKAGE / "reference_trainer_fixture.txt"), "--work", str(root),
                        "--fold", str(fold), "--out", str(root / "prior.json")]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(argv[0], run_name="__main__")
                old = json.loads((root / "prior.json").read_text())
                new = fit_prior(stats, {"A": 0, "B": 1}, fold)
                np.testing.assert_allclose(new["weights"], old["weights"], atol=1e-14, rtol=0)
                self.assertEqual(new["n_pairs"], old["n_pairs"])


class PipelineTests(unittest.TestCase):
    def test_response_weighted_scoring_and_other_pair_labels(self):
        subject, item = {"normalized_name": "x"}, {"benchmark_id": "A"}
        pair = {"benchmark": "A", "subject_key": "key", "subject": subject,
                "acquisition": [(item, int(i % 2 == 0)) for i in range(40)],
                "evaluation": [(item, [0, 0, 0]), (item, [1])]}
        model = Predictor({"grid": [.3, .8], "weights": [.4, .6]})
        acquired, scores = evaluator.evaluate([pair], model, None, str(PACKAGE), [[0]])
        for n in evaluator.BUDGETS:
            s = sum(entry[1] for entry in acquired[0][:n])
            p = model.posterior_mean(s, n)
            self.assertAlmostEqual(scores[0][n], (3 * p * p + (p - 1) ** 2) / 4)

    def test_fixed_folds_reject_reference_mismatch(self):
        folds = {"n_folds": 2, "seed": 0, "folds": {"A": 0, "B": 1}}
        ref = {"args": {"n_folds": 2, "seed": 0, "n_draws": 12, "draw_limit": 1000}}
        settings = runner.validate_settings(ref, folds, [1, 2])
        self.assertEqual(settings["fold_seed"], 0)
        self.assertEqual(settings["snapshot_seeds"], [1, 2])
        with self.assertRaises(ValueError):
            runner.validate_settings(ref, {**folds, "seed": 1}, [0])

    def test_full_pipeline_preserves_originals_and_pairs(self):
        data = fixture()
        with tempfile.TemporaryDirectory(prefix="eb test ") as tmp:
            root = Path(tmp)
            for directory in ["work", "results", "submissions/eb_grid", "dist"]:
                (root / directory).mkdir(parents=True, exist_ok=True)
            fold_data = {"n_folds": 2, "seed": 0, "folds": {"A": 0, "B": 1}}
            (root / "work/folds.json").write_text(json.dumps(fold_data))
            for filename in data:
                (root / "work" / filename).write_bytes(b"synthetic: IO mocked")
            ref = {"args": {"n_folds": 2, "seed": 0, "n_draws": 2, "draw_limit": 1000}}
            (root / "results/eb_grid.json").write_text(json.dumps(ref))
            shutil.copy2(PACKAGE / "reference_trainer_fixture.txt", root / "train_eb_prior.py")
            shutil.copy2(PACKAGE / "reference_model_fixture.txt", root / "submissions/eb_grid/model.py")
            (root / "submissions/eb_grid/prior.json").write_text("preserve original prior")
            (root / "dist/eb_grid_all.zip").write_bytes(b"preserve submission")
            originals = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}

            def run_training(command, **kwargs):
                with patch.object(sys, "argv", command[2:]), contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(command[2], run_name="__main__")

            args = argparse.Namespace(root=str(root), reference=None, seeds=[0, 1, 2], n_draws=None, methods=None)
            with patch("pandas.read_parquet", side_effect=parquet_reader(data)), \
                    patch.object(runner.subprocess, "run", side_effect=run_training), \
                    contextlib.redirect_stdout(io.StringIO()):
                output = runner.run(args)
            for path, content in originals.items():
                self.assertEqual(Path(path).read_bytes(), content)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "complete")
            tables = {name: pd.read_csv(output / f"{name}.csv") for name in runner.configurations(1000)}
            aligned = runner.align_results(tables)
            self.assertEqual(len(aligned), 9)
            self.assertEqual(set(aligned["eb_grid"].index.get_level_values("seed")), {0, 1, 2})
            zero = aligned["eb_zero_half"]
            np.testing.assert_allclose(zero["B0"], .25, atol=1e-14)
            np.testing.assert_allclose(zero[["B1", "B3", "B7", "B15", "B31"]],
                                       aligned["eb_grid"][["B1", "B3", "B7", "B15", "B31"]], atol=1e-14)
            for p in (output / "models").rglob("prior.json"):
                prior = json.loads(p.read_text())
                heldout = int(prior["fold"])
                self.assertTrue(all(fold_data["folds"][b] != heldout for b in prior["training_benchmarks"]))
            bad = {**tables, "eb_zero_half": pd.concat([tables["eb_zero_half"], tables["eb_zero_half"].iloc[:1]])}
            with self.assertRaises(ValueError):
                runner.align_results(bad)


if __name__ == "__main__":
    unittest.main()
