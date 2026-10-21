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

Download the **Starting Kit from [Codabench competition 17828](https://www.codabench.org/competitions/17828/)**. This is the download destination linked from the official competition page under [Package and submit](https://aims.stanford.edu/competition). Sign in with your own Codabench account if access requires it.

Use the competition's Starting Kit download, not a clone of the separate [GitHub baseline repository](https://github.com/aims-foundations/paiec_baseline). The latter has a different layout and smoke-test interface. The kit root required by this repository contains both files below:

```text
<kit-root>/
  tools/
    check_submission_zip.py
    run_smoke_test.py
```

The validation scripts used for the release are identified by their SHA-256 values in [`release/starting_kit.json`](../release/starting_kit.json). No archive version label or full-archive checksum was recorded, so those two file hashes are the available version identifiers. They do not identify every file in the kit. The kit is not redistributed here, and the current download has not been independently verified against these hashes.

For a new setup, save the downloaded ZIP outside this repository as `/workspace/paiec-starting-kit.zip` (or substitute its actual path). Extract it into a new directory; `mkdir` intentionally stops if that directory already exists:

```bash
bash -e <<'SH'
mkdir /workspace/paiec-kit-extracted
python -m zipfile -e /workspace/paiec-starting-kit.zip /workspace/paiec-kit-extracted
SH
```

From the root of this repository, run the following command after preparing `data/prepared`. It finds the kit root even if the ZIP has a top-level folder, verifies both script hashes, and connects the inputs only after they match. If using an existing preserved kit, set `extracted` to that kit's directory instead; there is no need to download or extract it again.

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

extracted = Path("/workspace/paiec-kit-extracted").resolve(strict=True)
expected = json.loads(Path("release/starting_kit.json").read_text())["files"]
required = {"check_submission_zip.py", "run_smoke_test.py"}
if set(expected) != required:
    raise SystemExit("Unexpected Starting Kit release record.")
roots = sorted({
    p.parent.parent
    for p in extracted.rglob("check_submission_zip.py")
    if p.parent.name == "tools"
    and (p.parent / "run_smoke_test.py").is_file()
})
if len(roots) != 1:
    raise SystemExit(
        "Expected exactly one Starting Kit root with both tools; found: "
        + repr([str(p) for p in roots])
    )
kit = roots[0]
for name, recorded_hash in sorted(expected.items()):
    actual = hashlib.sha256((kit / "tools" / name).read_bytes()).hexdigest()
    if actual != recorded_hash:
        raise SystemExit(
            f"Starting Kit version mismatch: {name}\n"
            f"recorded={recorded_hash}\nactual={actual}"
        )
    print(f"PASS: {name} matches the release record.")
subprocess.run([
    sys.executable, "tools/connect_inputs.py",
    "--work", "data/prepared", "--starting-kit", str(kit),
], check=True)
PY
```

If the download no longer contains the recorded scripts, retain the original kit or request the matching version from the competition organizers. Do not rename unrelated baseline tools or overwrite the recorded hashes to make a mismatch pass. Testing a newer kit is a separate compatibility check and should be recorded as such.

After connection:

```bash
python -m unittest discover -s tests -v
python -u prepare_eb_cap.py --root .
python tools/verify_reproduction.py
python tools/reproduce_cv.py --phase confirm
```

The connector installs the released fold assignment for newly prepared data. It refuses to replace a different existing assignment. Parquet reserialization can change file bytes; if data hashes differ, inspect the data contents and assess model-file equality and metric agreement separately.

The maintainer successfully rebuilt the submission from the preserved prepared dataset and existing kit on 2026-10-21; see [validation evidence](RESULTS.md#validation-scope). This does not verify reconstruction from a fresh dataset or kit download. Downloader network operations were tested locally with mocks. Actual access permissions and service availability depend on the execution environment.
