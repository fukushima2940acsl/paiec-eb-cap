# Mean-matched EB-Cap: results and limitations

Lower Brier ALC is better. Results below are maintainer-reported runs, with hosted
aggregate scores transcribed from screenshots and checked against supplied
formative summaries. This release does not claim an independent server rerun of
all experiments. The candidate rule was fixed before its full-data build and
the repeat submissions. Evaluation labels were not used to fit its mean target.

## Public local development comparisons

| Evaluation population | Old-only baseline ALC | Expanded ALC | Mean-matched ALC | Matched minus baseline | Exploratory 95% interval |
|---|---:|---:|---:|---:|---|
| Old benchmarks | 0.200066 | 0.200319 | 0.199753 | -0.000313 | [-0.000874, +0.000160] |
| Additional benchmarks | 0.212170 | 0.212705 | 0.211982 | -0.000188 | [-0.000420, +0.000089] |

Old-target evaluation: five fixed benchmark folds, seeds 3/4/5, 1,528 rows,
973 distinct subject–benchmark pairs and 30 benchmarks. Both training arms
excluded the held-out old fold; the expanded arm also excluded related new
benchmarks under the recorded overlap rules. Each target mean came only from
that fold's old training data.

Additional-target evaluation: five overlap-group folds, seeds 3/4/5, 2,071 rows,
1,183 distinct pairs, 60 sampled benchmarks and 57 sampled overlap groups.
Training excluded held-out groups and related old benchmarks. The excluded old
benchmarks had no cap-eligible pairs in the recorded run. Two eligible additional
benchmarks were never sampled. Absolute ALC levels across these two populations
are not directly comparable.

Intervals use 2,000 exploratory bootstrap resamples of benchmark clusters or
overlap components, respectively. Both intervals include zero. They do not
adjust for repeated comparisons, candidate selection or every dependence caused
by overlapping training sets. Seeds change sampling and acquired evidence, not
the benchmark folds. Repeated pairs are not independent new observations. These
public data had already been inspected during development; neither experiment
is an untouched final test.

Alternative weighting also changes the estimated effect. On the old population,
averaging repeated seeds within each pair first gives -0.000193, and equal
benchmark weighting gives -0.000137. On the additional population the respective
differences are -0.000207 and -0.000046. These are sensitivity analyses with
different estimands, not additional independent confirmations. On the old
population, the uncalibrated expanded model has a lower equal-benchmark ALC than
the mean-matched model, despite losing under the primary row-weighted analysis.

## Fixed-artifact hosted repeats, 27 October 2026

The primary descriptive comparison uses only the three new completed submissions
per method. Each completed submission receives equal weight; historical runs
are excluded. The intended order was baseline then mean-match, repeated three
times. Seven attempts were observed, including one cancellation and a later
baseline attempt. No best-run selection is used in the summary.

| ID | Time (JST) | Method | Status | Brier ALC |
|---|---|---|---|---:|
| 1010152 | 02:17 | EB-Cap | Finished | 0.203744 |
| 1010186 | 02:31 | Mean-match | Finished | 0.194683 |
| 1010242 | 02:47 | EB-Cap | Finished | 0.206213 |
| 1010259 | 02:54 | Mean-match | Finished | 0.167588 |
| 1010276 | 03:02 | EB-Cap | Cancelled | — |
| 1010465 | 03:53 | EB-Cap | Finished | 0.206785 |
| 1010647 | 04:27 | Mean-match | Finished | 0.182187 |

The maintainer suspects network trouble for 1010276. No usable score was
available in the supplied record; it is retained in the attempt history and
excluded from score summaries. Whether a score had been seen before cancellation
was not established. The analysis therefore describes the six completed runs,
not six uninterrupted attempts under the original protocol.

| Method | Completed runs | Mean ALC | Sample standard deviation | Range |
|---|---:|---:|---:|---|
| EB-Cap | 3 | 0.205581 | 0.001616 | [0.203744, 0.206785] |
| Mean-match | 3 | 0.181486 | 0.013561 | [0.167588, 0.194683] |

The observed difference of means is **-0.024095**. The standard deviations
describe run-to-run scores; they are not confidence intervals or standard errors.
No statistical-significance claim is made from three runs per method.

Each arm includes 26 pair observations but only 23 distinct pairs. Only **one
distinct pair** appears in both arms. This shared pair has ALC 0.087195 for EB-Cap
and 0.087987 for mean-match (+0.000792); one pair cannot establish general
superiority or inferiority. Anonymous IDs and private per-pair tables are omitted.

There is a direct diagnostic for sample-composition effects: both frozen
predictors return the same zero-label probability (within 1e-12), yet their
mean zero-label Brier scores are **0.245576** and **0.237973**, respectively.
That discrepancy is not an effect of the mean-matching algorithm. The fraction
of the overall observed difference attributable to method versus sampled
subjects cannot be identified from these runs.

The [official specification](https://aims.stanford.edu/competition) states that
formative evaluations redraw a hidden subset for each submission. Final automatic
evaluation uses a common hidden subset. Alternating submission order does not
make the formative comparison paired. The hosted result is useful as an external
execution and variability check; **it does not establish a 0.024095 algorithmic
improvement or predict a final leaderboard rank**.

## Historical results, excluded from the primary repeat comparison

| ID | Date (JST) | Method | Status | Brier ALC |
|---|---|---|---|---:|
| 1008013 | 2026-10-21 | EB-Cap | Finished | 0.185216 |
| 1010083 | 2026-10-27 | Mean-match | Finished | 0.210622 |

The two original runs had no shared subject–benchmark pair. Retaining their
opposite aggregate ordering avoids presenting only favorable observations.
Their results are context, not part of the prespecified new-three-versus-new-three
summary. No final competition submission ID is selected by this release.

## Reproduction and preservation

The canonical mean-match archive SHA-256 is
`5c2495ead27f0ec76e32567f07400de8a05442486e60ae05fb3c1a53e7fd7048`.
Its inference file matches the baseline exactly. Training, source and artifact
identifiers are in `release/`; the three generated data/environment records are
captured from the verified original server build before publication.

The original baseline code, old `release/` records, and public tag
`v0.1.0-eb-cap` remain unchanged. Local tools verify numerical inference and
artifact identity; hosted `Finished` status establishes completion of that hosted
run, not correctness of every scientific assumption or an improvement guarantee.
