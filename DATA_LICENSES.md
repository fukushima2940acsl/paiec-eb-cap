# Data licenses, credits and review scope

Updated 28 October 2026. Artifact-specific conditions remain under review.

## Scope of project and data licenses

The repository's MIT License covers the original project implementation and
documentation. It does not grant rights to third-party benchmark content or
override applicable terms for data-derived records.

Training data were obtained from aims-foundations/measurement-db at revision
54074453da7b9911be0f48e4c560be547bc84d21:
https://huggingface.co/datasets/aims-foundations/measurement-db

Credit: Nhi Truong, Sang T. Truong, and Sanmi Koyejo. The AI Measurement Data
Bank. AIMS Lab, Stanford University, 2026.
https://aimslab.stanford.edu/measurement-db

AIMS-owned curation contributions are licensed under CC BY-SA 4.0:
https://creativecommons.org/licenses/by-sa/4.0/
These contributions include metadata, organization and normalization. Individual
benchmarks retain their original terms. A public-release flag or inclusion in a
competition inventory does not itself grant a new redistribution license.

Our processing validates records, filters training pairs to 80–1,000 items,
computes pair statistics and benchmark histograms, fits priors, performs
training-only mean matching, and computes public-data CV scores. No endorsement
by AIMS or the original benchmark authors is implied.

## What the repository distributes

The repository excludes raw question text and item-level response tables.
It does distribute data-derived artifacts: training histograms, per-pair counts
and mean responses, fitted priors, and public-data CV score records. The frozen
NOTICE.md predates the CV addition; its wording about unbundled dataset contents
should be read with this explicit description of the later additions.

At commit 9dc1959, final mean-match training histograms cover 92 benchmarks;
the published training-statistics CSVs cover 156. These are different scopes.
No blanket redistribution clearance or automatic license for every derived
artifact is asserted here. Numerical summaries are not automatically subject
to every content license; applicable rights and access terms must be checked
for the particular material and use. Nor does this notice relicense all models,
software or statistics under CC BY-SA.

## Benchmark attribution review

DATA_ATTRIBUTIONS.json records current public metadata, upstream links,
requested citations and project usage. Its source metadata is pinned to GitHub
commit 652474f73fed42ebfe08690a7d48e87a78b78dd4. All 160 reviewed metadata
files now match content retrieved from the experiment-recorded Hugging Face
revision 54074453da7b9911be0f48e4c560be547bc84d21. The checks cover all 92
final-training benchmarks and all 156 benchmarks with published training
statistics. This is a metadata content check, not a new raw-data validation.

[METADATA_VERIFICATION.json](METADATA_VERIFICATION.json) records the evidence:
159 collected files were directly hashed and compared; live_agent_risk was
checked through an author-run server hash-only receipt whose revision, hash and
size match the reviewed GitHub file. This one file was initially excluded by
credential-pattern screening. Its downloaded contents were not included in the
receipt. It contributes to public statistics but not the final training fit.
The fixed-version README also confirms the recorded citation and the distinction
between AIMS-owned curation and upstream benchmark terms. Its prose differs from
the reviewed GitHub README.

These files were newly retrieved for the review on 28 October 2026. The match
does not establish what access conditions the author accepted at the time of
the original downloads, and it does not resolve artifact-specific upstream
conditions. The inventory remains a review aid, not blanket rights clearance.
Its metadata-derived content carries the AIMS credit and CC BY-SA notice in
that file. Unknown or absent license information is neither permission nor
proof that use is prohibited. Preserve upstream credits, notices and terms
where applicable, including dataset-specific conditions beyond the Data Bank.

Current unresolved examples include the conflicting license labels on Legal
RAG Bench, custom AgentRewardBench terms for WorkArena/VisualWebArena data,
and unclear redistribution terms reported for several upstream datasets.
Resolve these against the applicable source version and access conditions.

## Baseline and Starting Kit credit

The official empirical-mean baseline is described in PAIEC Predictor Baselines:
Sang Truong, Wale Salaudeen, Nhi Truong, Serena Wang, Angelina Wang,
Luke Guerdan, Ziang Xiao, Xing Xie, Nick Haber, and Sanmi Koyejo (2026).
https://github.com/aims-foundations/paiec_baseline

That repository's software is MIT licensed. If substantial code is copied or
adapted from it, retain its copyright and permission notice in the applicable
distribution; a bibliographic citation alone does not replace that notice.
The official Starting Kit is distributed separately and is not included here.
The baseline repository's license should not be assumed to license the Kit.
