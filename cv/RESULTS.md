# CV results and limits

Primary differences are mean-matched minus old-data-only ALC; negative is better.
One fixed prior is used across budgets, with ordinary Bayesian updating.

| Population | Old-only | Expanded | Mean-matched | Difference | Exploratory 95% interval |
|---|---:|---:|---:|---:|---|
| Old | 0.200066 | 0.200319 | 0.199753 | -0.000313 | [-0.000874, +0.000160] |
| Additional | 0.212170 | 0.212705 | 0.211982 | -0.000188 | [-0.000420, +0.000089] |

The source tables are unrounded in `records/*/expected/`. Both primary intervals
include zero. They use 2,000 cluster-bootstrap resamples with RNG seed 20261021,
clustering old results by benchmark and additional results by overlap component.
They do not adjust for repeated comparisons, selecting this candidate after
earlier development, all cross-benchmark overlap or shared training sets across folds.

Mean matching gives negative ALC differences for all three sampling seeds in both
populations, but not for every fold. Seeds vary sampled pairs and acquired labels,
not the benchmark folds. Repeated seed appearances are not independent new pairs.
The old-target B1 and B3 scores improve most. On additional benchmarks, B1 worsens
slightly while later budgets improve. B0 matches the old-only baseline by design.
Lower ALC does not imply improvement at every budget.

| Aggregation | Old difference | Additional difference |
|---|---:|---:|
| Primary: average evaluation rows | -0.000313 | -0.000188 |
| Average seeds within pair first | -0.000193 | -0.000207 |
| Equal weight per benchmark | -0.000137 | -0.000046 |

These weightings answer different questions. Under equal-benchmark weighting on
the old population, uncalibrated expansion (0.207070) is better than mean matching
(0.207355), despite the opposite ordering in the primary comparison. The
mean-matched versus uncalibrated-expanded contrast is secondary and cannot by
itself justify replacing the old-data baseline.

The same public benchmarks were already used in development. Neither comparison
is an untouched final test. Equal B0 predictions constrain the prior mean but
do not hold variance or tails fixed; the experiment is not a pure mean-only
intervention. Model predictions ignore item text and assume exchangeable labels.
Additional data can shift the benchmark mixture without helping this evaluation.

## Reproduction evidence

The archived runs are `meanmatch_old_ouoylcdj` and `meanmatch_h05_goy4`.
They reproduce preceding controls `screen_5znmvkts` and `newbench_16hova27`.
Server and separate local checks of the original evidence refitted the training
priors and all eleven aggregate tables per population. Maximum aggregate
discrepancy was below 2e-16, with mean errors below 1e-13. Published replay checks
use tolerance 1e-12 to avoid claiming bitwise portability for floating arithmetic.
`reproduce.py` also verifies all 528 posterior count states per fitted prior.

This is computational reproducibility of the recorded comparison, not an
independent replication of its statistical conclusion. No fresh downloading,
full raw-data scoring or new sampling is performed by the public replay command.
The source archive's excluded raw files were metadata-checked at collection;
their entire contents were not rehashed during that collection.

## Hosted comparisons remain separate

The [hosted results](../meanmatch/RESULTS.md) use mostly different evaluation pairs
across submissions. Their larger mean difference is not an identified method
effect. Those private per-pair records are not included in this CV package.
Successful numerical replay does not imply a final competition rank or a
guaranteed improvement on unseen data.
