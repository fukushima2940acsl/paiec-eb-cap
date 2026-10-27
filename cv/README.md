# Verified public CV records for EB-Cap mean matching

This addition reproduces the two principal local development comparisons from
saved public training statistics and paired result rows. It preserves the
original EB-Cap baseline, the mean-matched submitted predictor, and the
`v0.1.0-eb-cap` tag. No new competition entry is selected.

## Reproduce

From the repository root, use the existing Python environment with NumPy and pandas:

```bash
python -B -m unittest discover -s cv/tests -v
python -B cv/check_release.py
python -B cv/reproduce.py
```

No GPU, network, raw dataset or Starting Kit is needed for this replay. It refits
all five folds' three priors from the saved training statistics before reading
score tables, verifies all 528 posterior count states per prior, and regenerates
eleven aggregate tables per population. These include the 2,000-resample cluster
bootstrap and secondary weighting analyses. It compares them with the archived
results and the reported six-decimal summaries. All numeric checks use absolute
tolerance 1e-12; recorded rounded summaries use 5.1e-7.

Every execution creates a new ignored `results/cv_replay_<suffix>/` directory.
Success requires `manifest.json` with `status: complete`, plus `old/CHECKS.json`
and `new/CHECKS.json` with `status: passed`. Failed attempts remain marked failed;
the program never replaces a previous run or loosens a comparison automatically.
`python cv/reproduce.py --out results/cv_checks` selects another ignored output
parent. Do not put generated outputs inside `cv/`.

The original environment used Python 3.11.13, NumPy 2.3.2 and pandas 3.0.6.
The archived replay also passed under Python 3.12.14, NumPy 2.3.5 and pandas 2.2.3.
`requirements.txt` records the original numerical-library versions for a separate
environment if needed; installing them into an existing shared environment is
not a prerequisite. Each new replay records the versions actually used.

## Comparisons

| Population | Rows | Unique pairs | Benchmarks | Bootstrap clusters |
|---|---:|---:|---:|---:|
| Old benchmarks | 1,528 | 973 | 30 | 30 benchmarks |
| Additional benchmarks | 2,071 | 1,183 | 60 | 57 overlap components |

Both use five frozen folds, seeds 3/4/5, 60 draws per fold, and a 1,000-item draw
limit. Budgets are 0, 1, 3, 7, 15 and 31; Brier ALC weights are
0.1, 0.2, 0.2, 0.2, 0.2 and 0.1. Arms share evaluation pairs, acquired labels
and evaluation counts within each original experiment. These two evaluation
populations are different; their absolute ALC levels are not directly comparable.

| Population | Old-only ALC | Expanded ALC | Mean-match ALC | Mean-match minus old-only | Exploratory 95% interval |
|---|---:|---:|---:|---:|---|
| Old | 0.200066 | 0.200319 | 0.199753 | -0.000313 | [-0.000874, +0.000160] |
| Additional | 0.212170 | 0.212705 | 0.211982 | -0.000188 | [-0.000420, +0.000089] |

Lower is better. Both primary intervals include zero. Replaying the calculations
does not remove development-selection bias or establish improvement on unseen
benchmarks. See [RESULTS.md](RESULTS.md) for interpretation and limitations.

## What this release contains

- `records/{old,new}/pair_stats.csv` and `new_pair_stats.csv`: original public
  pair-level training statistics, preserved byte for byte.
- `folds.json` or `split.json`: the exact benchmark/group memberships and exclusions.
- `pairs/`: all three arms' public local evaluation rows. The display-only
  subject-name column was removed; numeric field strings, keys and row order
  are preserved. Keys come from the public-data preparation, not hosted feedback.
- `reference_controls/`: the preceding runs' two control tables for exact alignment.
- `priors.json`, `prior_summary.csv`, `evaluation_coverage.csv`: frozen fits,
  actual mean targets and evaluation coverage.
- `expected/`: eleven original aggregate CSVs per population.
- `run_record.json`: selected original protocol, snapshot signatures and control
  checks. It is a derived record with explicit omissions, not the original manifest.
- `source/`: byte-identical training, calibration, prediction and comparison
  functions from the verified runs. `reproduce.py` supplies portable replay paths.
- `PROVENANCE.json` and `SOURCE_HASHES.json`: original archive/member identities,
  transformations and public file hashes.

Training statistics include public benchmark names and subject keys to preserve
fold membership and paired comparisons. No original item content, raw response
rows, private Codabench feedback, credentials or absolute server paths are included.
See [DATA.md](DATA.md) for schemas, parsing rules and source attribution.

## Verified scope

The source archive was collected and independently reaggregated on 27 October
2026. Its checksum and all 392 inventoried member hashes matched. The server and
separate local replay reproduced the priors and all aggregate tables; table
discrepancies were below 2e-16, and B0 mean errors below 1e-13. Each original
manifest records 15 matching snapshot signatures and 30 control-reproduction checks.
The source collection recorded unchanged HEAD, 69 tracked files and the baseline
tag. Public export equivalence is checked again when preparing this release.
`public_replay_verification.json` records the successful replay of this exported
package, its wrapper/checker hashes, and the exact column-projection checks.

This command does **not** reconstruct the original sampling snapshots or rerun
scoring from raw responses. It checks recorded signatures, refits priors and
reaggregates stored public scores. Fresh downloading/preprocessing, full raw-data
scoring, all earlier exploratory methods and hosted resampling are separate tasks.
The original server paths and optional cache-based full-CV commands remain in the
maintainer's private evidence archive; no non-portable command is presented here
as a fresh-clone full evaluation recipe.

The [final-model reproduction](../meanmatch/README.md) and
[original baseline reproduction](../README.md#reproducing-a-published-release)
remain separate workflows. CV fold priors must not be substituted for full-data
submission priors.
