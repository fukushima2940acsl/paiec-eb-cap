# Results recorded on 2026-10-21

These tables summarize experiments reported from the execution server. Raw datasets, raw logs, and private evaluation-pair identifiers are not bundled. Distinguish these historical results from new results obtained by rerunning the published code.

## Confirmation runs

Five fixed benchmark folds, sampling seeds 3, 4, and 5, and `n_draws=60`. Every candidate uses the same sampled pairs and acquired labels. There are 1,528 evaluation rows, 973 distinct pairs, and 30 benchmarks. Repeated pairs across seeds are not independent observations.

| Method | ALC | Difference: candidate − eb_grid | Exploratory 95% interval | Fraction of evaluation rows won |
|---|---:|---:|---|---:|
| eb_grid | 0.201959 | 0 | — | — |
| eb_cap | 0.200066 | −0.001893 | [−0.006224, +0.002243] | 0.543194 |
| beta11 | 0.200974 | −0.000985 | [−0.004240, +0.001875] | 0.518979 |

| Seed | Evaluation rows | eb_grid | eb_cap | eb_cap difference | beta11 |
|---|---:|---:|---:|---:|---:|
| 3 | 509 | 0.201080 | 0.198678 | −0.002401 | 0.199752 |
| 4 | 506 | 0.201340 | 0.199768 | −0.001573 | 0.200367 |
| 5 | 513 | 0.203442 | 0.201737 | −0.001705 | 0.202787 |

| Method | Fold 0 difference | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| eb_cap | −0.003253 | −0.002627 | −0.000262 | −0.000588 | −0.001195 |
| beta11 | −0.003291 | −0.000820 | +0.000582 | +0.000942 | −0.000194 |

| Method | B0 | B1 | B3 | B7 | B15 | B31 |
|---|---:|---:|---:|---:|---:|---:|
| eb_grid | 0.253240 | 0.226912 | 0.202418 | 0.186751 | 0.179488 | 0.175212 |
| eb_cap | 0.246043 | 0.222891 | 0.201082 | 0.186292 | 0.179426 | 0.175235 |
| beta11 | 0.250000 | 0.223826 | 0.201576 | 0.186904 | 0.179903 | 0.175325 |

eb_cap has lower average ALC than eb_grid in all three seeds and all five folds. Most of the difference appears at small label budgets; B31 is nearly unchanged. The intervals use 2,000 bootstrap replicates with benchmarks as clusters and include zero. They are not corrected for candidate selection or multiple comparisons. Development reused the same folds, so this is not a final evaluation on unused benchmarks.

## Initial screening

Sampling seeds 0, 1, and 2; 412 evaluation rows, 302 distinct pairs, and 28 benchmarks. The original `n_draws` setting is captured in `configs/reference_eval.json`. `mix25` mixes a 25% uniform component into the prior and updates it with the observations. `zero_half` sets the prediction to 0.5 only at zero acquired labels.

| Method | ALC | Difference | Exploratory 95% interval |
|---|---:|---:|---|
| eb_grid | 0.202044 | 0 | — |
| eb_cap | 0.199308 | −0.002736 | [−0.006149, +0.000726] |
| eb_cap_mix25 | 0.199442 | −0.002603 | [−0.005733, +0.000658] |
| eb_cap_zero_half | 0.199799 | −0.002245 | [−0.004896, +0.000397] |
| eb_cap_mix25_zero_half | 0.199834 | −0.002210 | [−0.004824, +0.000420] |
| beta11 | 0.200655 | −0.001389 | [−0.004165, +0.001592] |
| eb_mix25_zero_half | 0.201160 | −0.000885 | [−0.001898, +0.000167] |
| eb_mix25 | 0.201442 | −0.000602 | [−0.001315, +0.000148] |
| eb_zero_half | 0.201655 | −0.000389 | [−0.000824, +0.000033] |

The cap-only configuration was selected for comparison with additional seeds. Those seeds assess stability to pair sampling and label acquisition.

## Codabench execution records

| Submission ID | File | Status | Displayed score | Submission time (JST) |
|---|---|---|---:|---|
| 1007671 | empirical_mean.zip | Finished | 0.193103 | 2026-10-21 16:44 |
| 1007899 | eb_grid_all.zip | Finished | 0.212039 | 2026-10-21 18:42 |
| 1008013 | eb_cap_all.zip | Finished | 0.185216 | 2026-10-21 19:33 |

These records come from the author's submission screenshots and feedback. Each formative submission uses a different sample, so this is not a controlled comparison. The paired local experiments above provide the main evidence for comparing eb_cap with eb_grid.

An earlier eb_grid submission, 1007670, reached prediction completion but failed during scoring. A subsequent submission finished; the failure cause was not established. The failed submission's displayed value is excluded from performance comparisons.

## Validation scope

Synthetic tests cover parity with the original training and inference code, fold isolation, pair identity, comparison alignment, submission packaging, failure handling, release-record consistency, and publication checks. Local tests mock Parquet reads and use stubs for official validation commands. Real-data reproduction and the actual official checks must also be run on the execution server.
