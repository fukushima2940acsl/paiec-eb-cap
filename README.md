# PAIEC: eb_cap

An empirical Bayes predictor for the **Predictive AI Evaluation Competition at NeurIPS 2026**. The method learns a prior over success rates from public training data, then updates it using the labels revealed during evaluation. It requires no GPU, embeddings, or external inference API.

This release preserves the `eb_cap` method associated with **Codabench submission 1008013** as a baseline for further experiments. Original code and documentation are released under the [MIT License](LICENSE). Training data and the official Starting Kit are obtained separately.

## Results

Lower Brier ALC is better. These numbers are from reported server runs. Release tooling was tested with synthetic fixtures; real-data reproduction must be checked separately using the commands below.

| Method | Confirmation-run ALC | Difference from eb_grid | Exploratory 95% interval |
|---|---:|---:|---|
| eb_grid | 0.201959 | 0 | — |
| eb_cap | **0.200066** | **−0.001893** | [−0.006224, +0.002243] |
| Beta(1,1) | 0.200974 | −0.000985 | [−0.004240, +0.001875] |

Five fixed benchmark folds and sampling seeds 3, 4, and 5 give 1,528 evaluation rows, 973 distinct subject–benchmark pairs, and 30 benchmarks. The benchmark bootstrap interval includes zero. These benchmarks were also used during development, so the results do not establish an improvement on unseen benchmarks.

Codabench submission **1008013 / eb_cap_all.zip / Finished / 0.185216** was recorded on 2026-10-21. Formative submissions use different sampled subsets; their aggregate scores alone do not provide a controlled comparison. See [experimental results](docs/RESULTS.md).

## Method

A subject is identified by its complete set of eight attributes. A *pair* means a subject and a benchmark.

1. Within each training pair, average repeated responses to each item, then average those item means to estimate the pair's success rate.
2. Fit the prior using **only pairs with 80 to 1,000 distinct items, inclusive**. Pairs with more than 1,000 items are excluded rather than truncated.
3. Round success rates to a 101-point grid on `[0,1]` using `numpy.rint`. Normalize each benchmark's histogram, sum the histograms with equal benchmark weights, add 0.001 to every bin, and normalize again.
4. Given `s` successes among `n` acquired labels for a pair, predict the posterior mean:

$$
\widehat p(s,n)=\frac{\sum_j w_j p_j^{s+1}(1-p_j)^{n-s}}
{\sum_j w_j p_j^s(1-p_j)^{n-s}}.
$$

The implementation clips grid endpoints to `[1e-6, 1-1e-6]` and computes posterior weights in log space. At zero labels it uses the learned prior mean. It does not replace this prediction with 0.5 or mix in a uniform prior. All items in the same pair receive the same probability. The only method change from `eb_grid` is the upper limit on training-pair size; both methods use the same submission `model.py`.

## Reproducing a published release

Use Python 3.10 or later, preferably the versions recorded in `release/environment.json`. Follow [data and environment setup](docs/DATA.md) to obtain the public training data and official Starting Kit, then connect them:

```bash
python tools/connect_inputs.py --work data/prepared --starting-kit /workspace/official-starting-kit
python -m unittest discover -s tests -v
python -u prepare_eb_cap.py --root .
python tools/verify_reproduction.py
```

Replace the example Starting Kit path with its actual location. The published repository must contain the `configs/` and `release/` records generated on the original execution server. The preparation ZIP does not contain those records yet; maintainers must complete the release-preparation step below before publication.

### Rebuild the submission

`prepare_eb_cap.py` trains on eligible pairs from all folds and creates a new directory under `dist/eb_cap_all_*/`. It checks inference parity for all 528 success-count/label-count combinations up to 31 labels, then runs the official ZIP checker and smoke test. It creates `eb_cap_all.zip` only after every check succeeds. The ZIP contains exactly `model.py` and `prior.json` at its root.

`verify_reproduction.py` compares both files byte for byte with the recorded hashes for submission 1008013. ZIP timestamps are not part of this comparison. If multiple archives exist, specify the newly generated one with `--zip dist/eb_cap_all_XXXX/eb_cap_all.zip`. Add `--data` to check full hashes of the connected training files as well.

Rebuilding is sufficient for this check; uploading an identical model to Codabench again is unnecessary. Submission inference uses only the Python standard library.

### Reproduce the comparisons

```bash
# eb_grid / eb_cap / Beta(1,1); seeds 3,4,5; n_draws=60
python tools/reproduce_cv.py --phase confirm

# Original nine-setting screen; seeds 0,1,2; recorded n_draws
python tools/reproduce_cv.py --phase screen
```

Each run writes a new `results/eb_ablation_*/` directory containing `comparison.txt`, a configuration manifest, and models for each fold. Cross-validation models are not full-data submission models.

Folds are defined at the benchmark level, and held-out benchmark labels are excluded from prior fitting. Candidates use the same sampled pairs and acquired labels. Budgets are `[0,1,3,7,15,31]`, with ALC weights `[0.1,0.2,0.2,0.2,0.2,0.1]`. The local evaluator approximates the documented sampling, acquisition, and scoring procedures; it is not guaranteed to be identical to the private production evaluator.

## Preparing the release from the original workspace

Keep the existing `/workspace/paiec_baseline` workspace and place this package in a separate `/workspace/paiec-eb-cap` directory. Run in the existing Python environment:

```bash
cd /workspace/paiec-eb-cap
python tools/finalize_repository.py --source /workspace/paiec_baseline
python -m unittest discover -s tests -v
python tools/check_public_source.py
```

`finalize_repository.py` validates the existing submission archive and build manifest, then creates reproducibility records in `configs/` and `release/`. Computing full dataset hashes may take time. It connects existing data and the Starting Kit through symlinks excluded from publication, and refuses to replace conflicting records. It does not install host software or upload anything.

If multiple distinct archives are found, explicitly select **the ZIP actually uploaded as submission 1008013**:

```bash
python tools/finalize_repository.py \
  --source /workspace/paiec_baseline \
  --submission-zip /workspace/paiec_baseline/dist/eb_cap_all_XXXX/eb_cap_all.zip
```

Replace `XXXX` with the actual directory name. The script does not retrieve the artifact from Codabench; the maintainer must confirm the association between the selected local ZIP and submission ID.

## Repository layout

| Path | Purpose |
|---|---|
| `build_dataset.py` | Prepare training and evaluation tables from public data |
| `experiments/pair_stats.py` | Item aggregation, pair eligibility, and prior fitting |
| `experiments/local_eval_v2.py` | Local evaluation with benchmark-level folds |
| `experiments/run_experiments.py` | Candidate configurations, fold training, and comparisons |
| `train_eb_prior.py` | Original eb_grid reference trainer |
| `submissions/eb_grid/model.py` | Submission inference shared by eb_grid and eb_cap |
| `prepare_eb_cap.py` | Full-data fitting and validated submission packaging |
| `tools/` | Provenance, reproduction, and publication checks |
| `tests/` | Synthetic tests of training, inference, packaging, and release tooling |
| `release/` | Data, environment, and artifact identifiers generated on the server |
| `PUBLIC_FILES.txt` | Explicit list of files to add to Git |

## Publication and further development

Follow the [publication guide](docs/PUBLISH.md) and preserve this baseline with the tag `v0.1.0-eb-cap`. Experiments can continue on a separate branch; see [development guidance](docs/DEVELOPMENT.md).

Claude and Codex assisted with implementation, validation, and documentation. See [provenance and AI assistance](NOTICE.md). The competition technical report and final submission process are separate deliverables.

References: [official competition page](https://aims.stanford.edu/competition), [official baseline repository](https://github.com/aims-foundations/paiec_baseline), and [public training data](https://huggingface.co/datasets/aims-foundations/measurement-db).
