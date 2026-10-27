"""Exact old-benchmark training function, copied from the expansion experiment."""
import json
import numpy as np
import pandas as pd
from vendor.pair_stats import fit_prior

METHODS = ("eb_cap", "eb_cap_expanded", "eb_cap_expanded_meanmatch")


def fit_priors(old_stats, new_stats, folds, fold):
    required = ['benchmark_id', 'subject_key', 'mean', 'n_items']
    for name, table in [('old', old_stats), ('new', new_stats)]:
        if table.empty or table[required].isna().any().any() or table.duplicated(required[:2]).any():
            raise ValueError('Invalid ' + name + ' pair statistics')
        if not np.isfinite(table['mean']).all() or not table['mean'].between(0, 1).all():
            raise ValueError('Invalid pair rates')
        if not table.n_items.ge(1).all() or not table.n_items.eq(table.n_items.astype(int)).all():
            raise ValueError('Invalid item counts')
    if not set(old_stats.benchmark_id) <= set(folds):
        raise ValueError('Old statistics contain unknown benchmarks')
    if set(new_stats.benchmark_id) & set(folds):
        raise ValueError('New benchmark identity overlaps the frozen baseline')
    def allowed(value):
        exclusions = json.loads(value)
        if not isinstance(exclusions, list) or any(type(f) is not int or f not in set(folds.values()) for f in exclusions):
            raise ValueError('Invalid fold exclusions')
        return fold not in exclusions
    if new_stats.groupby('benchmark_id').excluded_folds.nunique().gt(1).any():
        raise ValueError('Fold exclusions must apply to a whole new benchmark')
    extra = new_stats[new_stats.excluded_folds.map(allowed)].copy()
    extended_folds = {**folds, **{b: -1 for b in extra.benchmark_id.unique()}}
    base = fit_prior(old_stats, folds, fold, max_items=1000)
    expanded = fit_prior(pd.concat([old_stats[required], extra[required]], ignore_index=True),
                         extended_folds, fold, max_items=1000)
    return {'eb_cap': base, 'eb_cap_expanded': expanded}

