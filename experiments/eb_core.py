"""Standard-library-only EB inference; mixture weight is fixed before evaluation."""
import math


class Predictor:
    def __init__(self, prior, uniform_mix=0.0, zero_half=False):
        if not 0 <= uniform_mix <= 1:
            raise ValueError("uniform_mix must be in [0, 1]")
        grid, weights = prior["grid"], prior["weights"]
        if len(grid) != len(weights) or not grid:
            raise ValueError("invalid prior shape")
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in grid):
            raise ValueError("invalid prior grid")
        if not all(math.isfinite(w) and w > 0 for w in weights):
            raise ValueError("invalid prior weights")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-8):
            raise ValueError("prior weights must sum to one")
        self.grid = [min(max(p, 1e-6), 1 - 1e-6) for p in grid]
        self.log_w = [math.log(w) for w in weights]
        self.log_p = [math.log(p) for p in self.grid]
        self.log_q = [math.log(1 - p) for p in self.grid]
        self.mix = uniform_mix
        self.zero_half = zero_half
        self.cache = {}

    def posterior_mean(self, s, n):
        if not isinstance(n, int) or not isinstance(s, int) or not 0 <= s <= n:
            raise ValueError("require integer 0 <= s <= n")
        if (s, n) in self.cache:
            return self.cache[s, n]
        if (n == 0 and self.zero_half) or self.mix == 1:
            p = (s + 1) / (n + 2)
        else:
            logs = [lw + s * lp + (n - s) * lq
                    for lw, lp, lq in zip(self.log_w, self.log_p, self.log_q)]
            top = max(logs)
            ws = [math.exp(v - top) for v in logs]
            total = sum(ws)
            p = sum(v * w for v, w in zip(self.grid, ws)) / total
            if self.mix:
                # Prior mixture: (1-a) G_EB + a Beta(1,1).
                # Update component weights by their marginal likelihoods.
                log_eb = math.log1p(-self.mix) + top + math.log(total)
                log_uniform = (math.log(self.mix) + math.lgamma(s + 1)
                               + math.lgamma(n - s + 1) - math.lgamma(n + 2))
                m = max(log_eb, log_uniform)
                a, b = math.exp(log_eb - m), math.exp(log_uniform - m)
                p = (a * p + b * (s + 1) / (n + 2)) / (a + b)
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid prediction")
        self.cache[s, n] = float(p)
        return float(p)

    def predict(self, input, labeled=None):
        subject, item = input
        benchmark = item.get("benchmark_id", "")
        n = s = 0
        for (observed_subject, observed_item), label in labeled or []:
            if observed_subject == subject and observed_item.get("benchmark_id", "") == benchmark:
                if label not in (0, 1):
                    raise ValueError("acquired response must be binary")
                n += 1
                s += int(label)
        return self.posterior_mean(s, n)
