# Data, provenance and transformations

The upstream public source is
[aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db),
pinned at revision `54074453da7b9911be0f48e4c560be547bc84d21`.
The old collection had 61 catalogs. Of 99 additional catalogs attempted, 96 were
valid and three were quarantined. See the repository's [baseline data guide](../docs/DATA.md)
and [mean-match data guide](../meanmatch/DATA.md) for acquisition and validation.
Public availability is not a new license grant: upstream data retain their
respective terms. Original project software is covered by the repository's
[MIT License](../LICENSE); this addition does not relicense benchmark content.

## Training statistics

Each row has `benchmark_id`, `subject_key`, `mean` and `n_items`.
`mean` averages repeated trials within each item, then item means within a pair.
`n_items` is the number of distinct items. Only pairs with 80--1,000 items,
inclusive, enter the prior fit. Ineligible pairs are retained in the statistics
as in the original files; eligibility is applied by the preserved fitting code.
The additional-data statistics also retain `excluded_folds`, encoding overlap
exclusions for the old-target experiment.

These CSV files are copied byte for byte, including decimal strings. The
old-target old-statistics reader uses `float_precision='round_trip'`; the
additional-target reader retains its original default parsing behavior. Small
floating-point changes near `numpy.rint(mean * 100)` boundaries can change a bin,
so do not normalize CSV decimals or substitute a different parser silently.

Both initial priors are benchmark-uniform histograms on 101 points, with 0.001
smoothing added per bin after summing normalized benchmark histograms. The
expanded prior is exponentially tilted to the actual clipped-grid B0 prediction
of that fold's old-training prior. Held-out labels and scores do not enter that
solver. Tilting also changes variance and tail mass. The method was developed
after inspecting earlier results; training-only fitting does not erase that history.

## Evaluation rows and identifiers

`pairs/*.csv` and `reference_controls/*.csv` retain:

```text
seed, fold, benchmark, subject_key, n_eval_items, n_acquired,
B0, B1, B3, B7, B15, B31, ALC
```

The original display-only `subject` column is removed. Every remaining field
string and row order is preserved. `subject_key` is the existing public-data
identity key, not an anonymous private-evaluation ID. Benchmark names and keys
are retained for alignment and seed/benchmark weighting. No reidentification of
hidden competition targets is attempted. There are no raw labels, item texts,
private formative feedback tables or acquired-label sequences in this release.
Consequently, result rows support reaggregation but do not reconstruct raw scoring.

## Frozen splits and overlap

Old-target `folds.json` preserves the baseline benchmark folds. New training
benchmarks carry fold exclusions. Additional-target `split.json` preserves
the overlap graph, group assignments and common old/new exclusions. Any detected
normalized exact-content match or explicit family relation can join benchmarks;
ineligible benchmarks can bridge components. Related training benchmarks are
excluded before fitting. Exact matching cannot rule out paraphrases, translations,
image-only copies or unknown shared provenance.

`medhelm_head_qa` and `vl_rewardbench` were eligible but never sampled in the
additional-target run. The original snapshot inventory and coverage remain
unchanged. No extra benchmarks or seeds are added during replay.

## Provenance and redaction

The verified source archive is `cv_records_0widwn1a.zip`, SHA256
`57958821675ec1c5b536964107332ed29bd588c56617331368c224b4264f4236`.
`PROVENANCE.json` maps exported files to original archive member names, hashes
and explicit transformations. Byte-identical copies are distinguished from
column-projected CSVs. Public file identities are in `SOURCE_HASHES.json`.

`run_record.json` selects protocol, versions, completion flags, snapshot hashes,
control checks and limits from each original manifest. It records the original
manifest hashes while omitting absolute paths, command paths and nested raw-input
inventories. Its own bytes are new; the original hash does not attest to identical
bytes for this derived record. The private original archive remains preserved.
Snapshot signatures document earlier checks; this lightweight replay does not
recompute signatures from raw response data.
