# Data and reproduction scope

Both old and additional public training inputs use
[`aims-foundations/measurement-db`](https://huggingface.co/datasets/aims-foundations/measurement-db)
at revision **54074453da7b9911be0f48e4c560be547bc84d21**.
Follow upstream access requirements and terms. Authenticate in your own
environment; never put tokens in this repository.

The old baseline used 61 catalog directories listed in
`../release/data_source.json`. Additional directories are the frozen 99 entries
in `candidates.json`, with exact downloaded files in `release/data.json`.
Only benchmark catalogs, subjects, items and response tables are needed. Traces
and embeddings are not required by this addition.

## Lightweight reproduction

`release/training.json` stores integer histogram counts for each eligible public
training benchmark, separated into old and additional groups. Counts are derived
from the validated pair means before equal-benchmark normalization. These
sufficient statistics allow exact membership/weight reconstruction without raw
questions or subject-level response tables. Source licenses and data terms
continue to apply to derived material. `artifact/prior.json` is the fitted model
state already used for submission.

The release exporter requires the completed original build, validates its source
and saved inputs, and verifies that this histogram representation reproduces the
old baseline and fixed candidate. This is a replay of the preserved validated
cache, not an independent fresh-download audit.

## Full public-data path

First obtain and prepare the old data using the repository's existing
[baseline instructions](../docs/DATA.md). The prepared directory must contain
`responses.parquet`, `items.parquet`, `subjects.parquet`, and the unchanged
`folds.json`; normally it is connected as `work/`. This step uses the old frozen
baseline data list, not the expanded data list.

From the repository root, inspect the additional download capacity **before**
downloading. NumPy and pandas are required for the fitting utilities; PyArrow
is also needed for data validation. Install the existing `requirements-data.txt`
only if the Hugging Face client is not already available.

```bash
python -B meanmatch/download_data.py
```

Then choose a capacity bound based on that output. For example, the following
allows at most 5 GiB of additional downloads; it stops before downloading if the
recorded total exceeds this value or if the staging/scratch reserve is unavailable.
This is a limit, not a claim that every environment needs exactly 5 GiB.

```bash
python -B meanmatch/download_data.py --max-download-gib 5
python -B meanmatch/prepare_data.py \
  --old-prepared work \
  --new-data payload/meanmatch-additional \
  --out private
python -B meanmatch/reproduce.py --official-kit starting_kit
```

Previously downloaded additional files can be reused by pointing `--new-data`
at the original expansion experiment's `data/downloads/` directory. All 396
recorded table hashes are checked. Original data are never overwritten.
The raw preparation command creates a unique `private/meanmatch_data_*/` directory,
recomputes old and additional statistics, and requires exact agreement with the
published histogram counts and quarantine outcomes. It fails instead of changing
the released model to accommodate a mismatch. The following reproduction command
uses the published counts only after that agreement has been established.

## Validation and overlap handling

Additional catalogs must be public, binary and item-level. The validation checks
schema, identifiers, duplicate records, subject/item joins, binary non-missing
grades and usable item content. Repeated responses are averaged within item,
then items within a pair. Malformed catalogs are quarantined in full, before
model comparison. Missing responses are counted and omitted under the frozen
rule. The original validation implementation is retained, with only its import
of the eight subject fields moved to the equivalent local `constants.py`.

For old-target CV, normalized exact-content overlap and an explicit `helm_bbq` /
`bbq` family rule determined whole-benchmark exclusions from held-out folds.
The additional-target experiment used connected overlap groups. In the final
all-public-training fit there is no held-out public fold, so these CV exclusions
are not applied. All valid eligible public training benchmarks contribute.

The method does not merge or reweight overlapping additional benchmarks.
Paraphrases, translated copies, image-only overlaps and unrecognized shared
provenance can remain. Equal weight per benchmark is retained, so correlated
benchmark families may influence the prior more than independent families.

## What remains to be verified

The source package is tested on synthetic data and the actual frozen submission
artifact. Production-cache export and official checks run on the maintainer's
server. A complete fresh-download replay has not been claimed as completed;
record its result separately if performed. Parquet bytes may differ after
reserialization, whereas the final derived integer histograms must match.
Private test data and per-pair private feedback are not distributed.
