# Data and environment setup

## Capturing the original execution environment

For the release maintainer, `tools/finalize_repository.py` connects to the prepared data in `/workspace/paiec_baseline/work` and the existing `starting_kit` directory. It reuses the working environment and does not duplicate the dataset.

Generated records:

- `release/environment.json`: Python, numpy, pandas, and pyarrow versions.
- `release/data_fingerprints.json`: sizes and SHA-256 hashes of the five prepared input files.
- `release/data_source.json`: public dataset revision and download inventory from the original `corpus_manifest.json`. Missing revisions are recorded as null, not inferred.
- `release/starting_kit.json`: hashes of the official ZIP checker and smoke-test scripts used locally.
- `configs/folds.json`: the benchmark fold assignment actually used.
- `configs/reference_eval.json`: the original sampling settings, with local absolute paths omitted.

## Setting up an independent environment

Create a dedicated virtual environment if a suitable environment is not already available. Select the Python version recorded in `release/environment.json` for the `python3` command below.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -c 'import json; from pathlib import Path; p=json.loads(Path("release/environment.json").read_text())["packages"]; Path(".venv/requirements-recorded.txt").write_text("\n".join(k+"=="+v for k,v in p.items())+"\n")'
python -m pip install -r .venv/requirements-recorded.txt
```

If the recorded versions cannot be installed on the target platform, `requirements.txt` lists the general dependency ranges. Recheck numerical results and artifact equality when using a different environment.

### Public training data

Follow the terms and any access-approval process for [measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db). Keep authentication in your own environment; do not store tokens in this repository.

If `release/data_source.json` contains a revision and directory inventory:

```bash
python -m pip install -r requirements-data.txt
python tools/download_recorded_data.py --out payload/data
python build_dataset.py --data payload/data --out data/prepared
```

The downloader fetches the four required Parquet table types at the recorded revision. It does not fetch traces or embeddings. If the original download has no recorded revision, the author's preserved prepared dataset is needed. Downloading the latest dataset does not reproduce the historical experiment.

### Official Starting Kit

Obtain the official Starting Kit separately through the [competition page](https://aims.stanford.edu/competition) or [official baseline repository](https://github.com/aims-foundations/paiec_baseline). Place it outside this public repository. The required checks are `tools/check_submission_zip.py` and `tools/run_smoke_test.py`.

Official distributions may change. Compare the validation-script hashes with `release/starting_kit.json` and record differences. This repository does not include a complete snapshot of the Starting Kit.

Assuming the kit is available at `/workspace/official-starting-kit`:

```bash
python tools/connect_inputs.py --work data/prepared --starting-kit /workspace/official-starting-kit
python -m unittest discover -s tests -v
python -u prepare_eb_cap.py --root .
python tools/verify_reproduction.py
python tools/reproduce_cv.py --phase confirm
```

The connector installs the released fold assignment for newly prepared data. It refuses to replace a different existing assignment. Parquet reserialization can change file bytes; if data hashes differ, inspect the data contents and assess model-file equality and metric agreement separately.

Downloader network operations were tested locally with mocks. Actual access permissions and service availability depend on the execution environment.
