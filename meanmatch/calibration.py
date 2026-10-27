"""One training-only exponential tilt; no evaluation labels, budgets, or scores."""
from copy import deepcopy
import math

from vendor.eb_core import Predictor
from vendor.pair_stats import fit_prior

METHODS = ('eb_old_only', 'eb_expanded', 'eb_expanded_meanmatch')
MEAN_TOLERANCE = 1e-13


def validated(prior):
    Predictor(prior)  # Same validity and endpoint clipping as released inference.
    grid = [float(x) for x in prior['grid']]
    if len(grid) < 2 or any(a >= b for a, b in zip(grid, grid[1:])):
        raise ValueError('A strictly increasing grid with at least two points is required')
    q = [min(max(p, 1e-6), 1 - 1e-6) for p in grid]
    w = [float(x) for x in prior['weights']]
    total = math.fsum(w)
    return grid, q, [v / total for v in w]


def moments(prior):
    grid, q, w = validated(prior)
    mean = math.fsum(p * v for p, v in zip(q, w))
    return {'mean': math.fsum(p * v for p, v in zip(grid, w)),
            'prediction_B0': Predictor(prior).posterior_mean(0, 0),
            'effective_variance': math.fsum(v * (p - mean) ** 2 for p, v in zip(q, w)),
            'tail0': math.fsum(v for p, v in zip(grid, w) if p <= .02),
            'tail1': math.fsum(v for p, v in zip(grid, w) if p >= .98)}


def match_old_mean(expanded_prior, old_prior):
    """q is the clipped grid used by inference; w'_i proportional to w_i exp(lambda*q_i).

    The target is the old-training prior's actual zero-label prediction, not an
    average of evaluation responses or the raw (unclipped) reported grid mean.
    Lambda is a root of a monotone mean equation, not a validation hyperparameter.
    """
    grid, q, w = validated(expanded_prior)
    old_grid, _, _ = validated(old_prior)
    if old_grid != grid:
        raise ValueError('Old and expanded priors must use the same frozen grid')
    target = Predictor(old_prior).posterior_mean(0, 0)
    if not min(q) < target < max(q):
        raise ValueError('Target mean is not strictly inside the supported range')
    log_w = [math.log(v) for v in w]

    def tilt(value):
        logs = [v + value * p for v, p in zip(log_w, q)]
        top = max(logs)
        values = [math.exp(x - top) for x in logs]
        normalizer = math.fsum(values)
        weights = [v / normalizer for v in values]
        return math.fsum(p * v for p, v in zip(q, weights)), weights

    original = Predictor(expanded_prior).posterior_mean(0, 0)
    iterations, value = 0, 0.0
    if abs(original - target) <= MEAN_TOLERANCE:
        weights = list(expanded_prior['weights'])  # An identity operation stays byte-stable in its weights.
    else:
        lower, upper = -1.0, 1.0
        for _ in range(60):
            if tilt(lower)[0] <= target:
                break
            lower *= 2
        else:
            raise ValueError('Could not bracket target from below')
        for _ in range(60):
            if tilt(upper)[0] >= target:
                break
            upper *= 2
        else:
            raise ValueError('Could not bracket target from above')
        for iterations in range(1, 201):
            value = (lower + upper) / 2
            mean, weights = tilt(value)
            if abs(mean - target) <= MEAN_TOLERANCE:
                break
            if mean < target:
                lower = value
            else:
                upper = value
        else:
            raise ValueError('Mean constraint did not converge')
    if not all(math.isfinite(v) and v > 0 for v in weights):
        raise ValueError('Tilt underflowed or produced an invalid weight; no clipping fallback is allowed')
    result = deepcopy(expanded_prior)
    result['weights'] = weights
    result.update({k: v for k, v in moments(result).items() if k in ('mean', 'tail0', 'tail1')})
    achieved = Predictor(result).posterior_mean(0, 0)
    if abs(achieved - target) > 1e-12:
        raise ValueError('Released inference does not satisfy the mean constraint')
    result['calibration'] = {'method': 'exponential_tilt', 'lambda': value,
        'target_source': 'fold_old_training_prior_zero_label_prediction',
        'target_mean': target, 'original_mean': original, 'achieved_mean': achieved,
        'absolute_error': abs(achieved - target), 'iterations': iterations,
        'mean_tolerance': MEAN_TOLERANCE, 'evaluation_labels_used': False,
        'note': 'The variance and tail weights also change; this is not a pure mean-only intervention.'}
    return result


def refit_controls(old, new, split, fold):
    """Reproduce the preceding split's two training-only priors before calibrating."""
    row = split['folds'][fold]
    if row['fold'] != fold:
        raise ValueError('Fold ordering changed')
    base = old[~old.benchmark_id.isin(row['excluded_old_benchmarks'])].copy()
    extra = new[~new.benchmark_id.isin(row['excluded_new_benchmarks'])].copy()
    if set(old.benchmark_id) & set(new.benchmark_id):
        raise ValueError('Old/new benchmark identity collision')
    held = set(row['evaluation_benchmarks'])
    if held & (set(base.benchmark_id) | set(extra.benchmark_id)):
        raise ValueError('Evaluation benchmark leaked into training')
    for table in (base, extra):
        if not table.n_items.between(80, 1000).any():
            raise ValueError('No eligible training pairs after exclusions')
    import pandas as pd
    columns = ['benchmark_id', 'subject_key', 'mean', 'n_items']
    tables = {'eb_old_only': base[columns],
              'eb_expanded': pd.concat([base[columns], extra[columns]], ignore_index=True)}
    priors = {}
    for method, table in tables.items():
        prior = fit_prior(table, {b: 0 for b in table.benchmark_id.unique()}, 'all', max_items=1000)
        prior['evaluation_fold'] = fold
        prior['excluded_old_benchmarks'] = row['excluded_old_benchmarks']
        for group in split['groups']:
            if group['group'] in row['evaluation_groups'] and set(group['members']) & set(prior['training_benchmarks']):
                raise ValueError('Held-out overlap group leaked into training')
        priors[method] = prior
    if any(priors['eb_expanded']['training_pair_counts'].get(b) != count
           for b, count in priors['eb_old_only']['training_pair_counts'].items()):
        raise ValueError('Old training composition differs between arms')
    return priors
