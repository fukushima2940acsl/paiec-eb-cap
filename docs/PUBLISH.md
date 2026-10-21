# Publishing the repository on GitHub

Use the dedicated `/workspace/paiec-eb-cap` directory rather than turning the original experiment workspace into a public repository. Raw data, local results, submission archives, and credentials are excluded from the publication list. The following commands are intended for the release maintainer.

## 1. Capture the existing experiment records

Place the preparation ZIP at `/workspace/paiec_eb_cap_public_en_20261021_v2.zip` and run inside the existing experiment container. If the destination already exists, inspect it first instead of extracting a full package over it.

```bash
cd /workspace
python -m zipfile -e paiec_eb_cap_public_en_20261021_v2.zip .
cd /workspace/paiec-eb-cap
python tools/finalize_repository.py --source /workspace/paiec_baseline
python -m unittest discover -s tests -v
python tools/check_public_source.py
```

If multiple distinct submission archives are found, pass `--submission-zip` with the path actually uploaded as submission 1008013. The script stops if current data or relevant source files differ from the original build record. Investigate a mismatch before modifying any records.

For an already extracted preparation package, the optional documentation-only archive can be applied from `/workspace`:

```bash
cd /workspace
python -m zipfile -e paiec_eb_cap_docs_en_20261021.zip .
```

This replaces six Markdown documents with their English versions. It does not change source code, generated release records, or connected data. Review any local documentation edits before applying it.

## 2. Rebuild in the public repository directory

```bash
cd /workspace/paiec-eb-cap
python -u prepare_eb_cap.py --root .
python tools/verify_reproduction.py
```

Confirm that both `model.py` and `prior.json` match the recorded hashes. If multiple ZIPs are present, use `--zip` to select the output path printed by the builder. When compute time is available, rerun the confirmation comparison:

```bash
python tools/reproduce_cv.py --phase confirm
```

Compare its values with `docs/RESULTS.md`. If they differ, examine environment, data, and fold records and document the discrepancy.

## 3. Create the local Git history

If Git author name and email are not configured, set them for this repository only. A GitHub noreply address may be used. There is no need to change shared global Git settings.

```bash
cd /workspace/paiec-eb-cap
git init -b main
git add --pathspec-from-file=PUBLIC_FILES.txt
python tools/check_public_source.py --staged
git diff --cached --stat
git diff --cached
git commit -m "Release reproducible eb_cap baseline"
git tag -a v0.1.0-eb-cap -m "eb_cap baseline for submission 1008013"
```

`PUBLIC_FILES.txt` lists the files to add explicitly. The staged check reads the actual Git index, rejects unlisted files, and checks common credential formats. It cannot detect every possible secret, so review the diff as well. The explicit file list avoids the need for `git add .` or forced additions of data directories.

## 4. Create an empty public GitHub repository and push

In GitHub's **New repository** form, create a repository such as `paiec-eb-cap` with **Public** visibility. Do not initialize a README, .gitignore, or license on GitHub; they are already present in the local commit.

Enter the SSH or HTTPS repository URL shown by GitHub. Do not embed a password or token in that URL.

```bash
printf 'GitHub repository URL: '
read -r PAIEC_REPO_URL
git remote add origin "$PAIEC_REPO_URL"
git push -u origin main
git push origin v0.1.0-eb-cap
```

These push commands publish the repository. Use your usual GitHub authentication method. After publication, verify that the README, MIT License, release records, and tag are visible, and that raw data and local logs are absent.

Preparing this package alone does not publish it. The competition technical report and final submission process are separate requirements. Consult the [official competition page](https://aims.stanford.edu/competition) for current instructions.

Reference: [GitHub's guide to adding locally hosted code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).
