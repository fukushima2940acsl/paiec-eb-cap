# Continuing development from the preserved baseline

Use `v0.1.0-eb-cap` to preserve the method associated with submission 1008013. Do not move that tag or overwrite its release records. Create a branch for subsequent experiments:

```bash
git switch -c experiment/next-improvement
```

Compare new candidates with eb_cap using the same sampled pairs and acquired labels. Use Brier ALC as the primary metric, and retain differences by budget, fold, and seed. Record the configuration and output directory for each experiment. Add new source files deliberately to `PUBLIC_FILES.txt` before publishing them.

If a candidate is selected, fit it on the full eligible training data and create a separate submission artifact. `prepare_eb_cap.py` is intended to reproduce the preserved eb_cap baseline; keep builders for new methods distinct.

Repeated development on the same folds introduces selection bias. Additional seeds test stability to sampling and label acquisition; they do not constitute an independent evaluation on unseen benchmarks. Do not select a method solely from differences in aggregate Codabench formative scores.

Preserve improvements with new tags and result records. Publishing this baseline does not prevent further experiments or submissions.
