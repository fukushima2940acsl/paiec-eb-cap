# EB-Cap with a fixed mean-matched expanded prior

This addition preserves the original `eb_cap` baseline and tag `v0.1.0-eb-cap`.
It changes the learned prior, while using the **byte-identical baseline inference
code** at `submissions/eb_grid/model.py`. Inference requires only Python's standard
library, no GPU and no external API. Acquisition remains random; no `labeling.py`
is supplied. New documentation and tooling are in English. Frozen source files
retain their original comments and bytes where needed for artifact identity.

The candidate was reported as successfully submitted under IDs **1010083,
1010186, 1010259 and 1010647**. These associations come from the maintainer's
submission records; the hosted ZIPs were not independently downloaded.

## Fixed method

1. Aggregate repeated responses within each item, then items within each
   subject–benchmark pair. Retain pairs with **80–1,000 distinct items inclusive**;
   exclude larger pairs rather than truncating them.
2. Quantize each pair mean onto a 101-point grid using `numpy.rint`. Normalize
   each benchmark's count histogram, sum benchmarks with equal weight, add
   0.001 per grid bin, then normalize. This is the unchanged EB-Cap estimator.
3. Fit the old-only prior on **5,437 pairs / 30 benchmarks**, and the expanded
   prior on **10,187 pairs / 92 benchmarks**. Additional data contribute 4,750
   pairs / 62 benchmarks. Of 99 additional catalogs, 96 passed validation and
   three were quarantined in full; not every valid catalog has eligible pairs.
4. Tilt the expanded weights to match the actual zero-label prediction of the
   old-only prior:

```text
q[j] = clip(grid[j], 1e-6, 1 - 1e-6)
target = posterior_mean(old_training_prior, successes=0, labels=0)
w_match[j] = w_expanded[j] * exp(lambda * q[j]) / Z
sum_j q[j] * w_match[j] = target
```

`lambda` solves this mean constraint by the frozen bisection rule; it is not
selected using validation scores or private evaluation labels. The constraint
uses the **clipped inference grid**, not the raw grid mean. The final target is
approximately **0.4698848889308734**, computed from all old public training data.
Every budget uses the same calibrated prior and the original Bayesian update.
Tilting changes variance and tail mass as well as the mean.

`calibration.py` is byte-identical to the previously tested rule. Its historical
metadata field `target_source` contains the word `fold`; it is retained unchanged
for artifact identity. In this full-training release the target uses all old
training folds, not a CV fold or an average of CV targets.

## Reproduce the fitted submission

From the repository root, use the existing environment or follow the baseline
[environment setup](../docs/DATA.md). `release/environment.json` **inside this
directory** records the original mean-match build versions and export versions.
The baseline's root-level `release/environment.json` remains unchanged.

```bash
python -B -m unittest discover -s meanmatch/tests -v
python -B meanmatch/check_release.py
python -B meanmatch/reproduce.py --official-kit starting_kit
```

The official kit must match the baseline's `release/starting_kit.json`; obtain it
as described in the baseline data guide. To perform the numerical and artifact
checks without an available kit, omit `--official-kit`; the run manifest then
explicitly records that official checks were not run.

The normal reproduction path:

- Refits old and expanded priors from committed **public-training histogram
  counts**, then applies the frozen mean constraint. These counts are sufficient
  statistics for this quantized estimator; they contain no subject identifiers,
  question text or private evaluation responses.
- Verifies the original baseline prior's canonical bytes against its recorded
  SHA-256, then checks both refitted priors numerically against their canonical
  references: exact training composition, tolerance 1e-14 for grid/weights and
  summary moments, and 1e-12 for all 528 posterior count states. Numerical replay
  is distinct from file identity: the baseline JSON omits a trailing newline,
  while new records include one. Last-bit floating-point differences are allowed
  within the original builder's tolerance; substantive model changes are rejected.
- Checks the actual submission's 1,056 predictions, including evidence from
  unrelated subjects and benchmarks, and repeats the check after extraction.
- Packages the preserved canonical `prior.json` and original `model.py` bytes
  **after** successful refit verification. The numerical refit and canonical
  byte packaging are distinct checks; no discrepant fit is silently replaced.
- Runs the preserved official ZIP check and smoke test when a kit is supplied.
- Writes a new ignored `dist/meanmatch_*/` directory and checks the full ZIP hash.
  It does not change the baseline, Git refs or prior builds.

An explicit `--artifact-only` mode checks inference and packaging without
training refit. It is useful for tooling QA but is **not** training reproduction.
All successful manifests record which checks were performed. ZIP compression
differences across runtimes can cause the whole-archive check to stop even when
members agree; use the recorded environment and inspect the mismatch.

## Recompute from public raw data

See [DATA.md](DATA.md) for the pinned snapshot, capacity inspection, download and
validation commands. The raw-to-histogram path is provided separately from the
lightweight sufficient-statistic replay. Neither a fresh complete download nor
the full production dataset can be run on the assistant's local machine; do not
interpret synthetic tests or inference checks as evidence of that full run.
The maintainer's export verifies the preserved completed build and cached
training statistics. Raw-data replay reports success only after matching the
released histograms and quarantine decisions.

## Results and scope

See [RESULTS.md](RESULTS.md) and the machine-readable aggregate records in
`release/reported_results.json`. Local development results show small mean
improvements with intervals including zero. The three new hosted runs averaged
0.205581 for EB-Cap and 0.181486 for mean matching, but used largely different
evaluation pairs. **The observed difference is not a causal estimate of method
improvement.** Raw private feedback tables and anonymous identifiers are not
distributed.

This package reproduces the final training estimator and submitted predictor.
It records earlier local CV summaries, but does not claim that all exploratory
experiments or the grouped additional-benchmark CV are rerun by `reproduce.py`.
Hosted resampling cannot be replayed locally from the published code alone.

The separate [CV records package](../cv/README.md) now reproduces both principal
local comparisons from saved training statistics and paired public result rows.
It refits fold-specific priors and replays the aggregate tables, including their
intervals. It does not replace the final-model `reproduce.py` workflow or claim
a new raw-response scoring run.


## Maintainer: capture the original verified build

This step is needed once when adding this directory to the baseline repository.
Consumers of a completed published release already have the generated records.

```bash
python -B meanmatch/export_release.py \
  --builder /workspace/paiec_eb_meanmatch_submit_20261025
```

It selects only a completed build whose ZIP has the fixed submitted hash. An
explicit `--build /absolute/path/to/builds/meanmatch_all_SUFFIX` can select the
same artifact. It verifies the original builder source, invokes its saved-build
verifier, refits the histogram representation, and exports three records:

- `meanmatch/release/training.json`: public-training histogram counts.
- `meanmatch/release/data.json`: public repository paths, pinned revision,
  download sizes/hashes and validation outcomes.
- `meanmatch/release/environment.json`: versions and verification scope.

The preserved `artifact/eb_cap_prior.json` is an exact copy of the original
baseline's submitted prior, used only as a verification reference. It does not
replace the baseline or the candidate. Export and reproduction manifests record
the observed maximum prediction error against this reference.

Original manifests, absolute server paths, credentials, raw responses, and
private formative feedback are not copied. A conflicting existing record stops
the exporter. The original experiment workspace is read only.

Claude and Codex assisted with development; Codex assisted with this reproduction
addition. See the repository [NOTICE](../NOTICE.md). The MIT code license does not
replace upstream dataset licenses or access terms.
