"""ノンパラメトリック経験ベイズによる予測。

学習データから推定した「ペアの正答率 p の分布」（prior.json、[0,1] 上のグリッド）を事前分布とし、
この subject×benchmark で獲得したラベル（正解 s 件 / n 件）で事後分布を求め、その平均を返す。

    事後の重み ∝ 事前の重み(p) × p^s × (1 - p)^(n - s)
    予測 = Σ p × 事後の重み

  - ラベルなし（予算 0）：事前分布の平均（学習データ全体の典型的な正答率）。
  - 全問正解型・全問不正解型のペアが学習データに多ければ、少ない連続正解でも 1 に近づく。
  - 普通のペアなら、1 件の結果では大きく動かない。
"""

import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "prior.json")) as _f:
    _PRIOR = json.load(_f)

GRID = [min(max(p, 1e-6), 1.0 - 1e-6) for p in _PRIOR["grid"]]
LOG_W = [math.log(w) for w in _PRIOR["weights"]]
LOG_P = [math.log(p) for p in GRID]
LOG_Q = [math.log(1.0 - p) for p in GRID]


def posterior_mean(s: int, n: int) -> float:
    logs = [lw + s * lp + (n - s) * lq for lw, lp, lq in zip(LOG_W, LOG_P, LOG_Q)]
    top = max(logs)
    ws = [math.exp(v - top) for v in logs]
    return sum(p * w for p, w in zip(GRID, ws)) / sum(ws)


_CACHE = {}


def predict(input, labeled=None):
    subject, item = input
    benchmark = item.get("benchmark_id", "")
    n = s = 0
    for (obs_subject, obs_item), label in labeled or []:
        if obs_subject == subject and obs_item.get("benchmark_id", "") == benchmark:
            n += 1
            s += label
    key = (s, n)
    if key not in _CACHE:
        _CACHE[key] = posterior_mean(s, n)
    return float(_CACHE[key])