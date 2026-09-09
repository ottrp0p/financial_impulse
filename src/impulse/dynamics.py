"""Level-conditioned drift mu(x) and volatility sigma(x).

Nadaraya-Watson kernel regression of the one-bar forward log-return on the
level it started from (Florens-Zmirou / Bandi-Phillips style nonparametric
diffusion estimation, in per-bar units).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bars import Bars
from .grid import PriceGrid, decay_weights


def local_dynamics(bars: Bars, grid: PriceGrid, tau: float | None = None,
                   bandwidth: float | None = None) -> pd.DataFrame:
    """Returns DataFrame indexed by level with columns mu, sigma, n_eff.

    mu(x)    = sum_i w_i(x) r_i / sum_i w_i(x)           (per-bar drift)
    sigma(x) = sqrt( sum_i w_i(x) r_i^2 / sum_i w_i(x) - mu^2 )
    n_eff(x) = (sum w)^2 / sum w^2  (Kish effective sample size at that level)
    with w_i(x) = K((x - x_i)/h) * decay_i, x_i = log close at bar i, r_i = x_{i+1} - x_i.
    """
    h = grid.bandwidth if bandwidth is None else bandwidth
    x = bars.log_close
    r = np.diff(x)
    x0 = x[:-1]
    d = decay_weights(len(x0), tau)
    # kernel matrix (n_levels, n_obs); chunk if large to bound memory
    mu = np.empty(grid.n)
    sig = np.empty(grid.n)
    neff = np.empty(grid.n)
    chunk = max(1, int(2e7 // max(len(x0), 1)))
    for a in range(0, grid.n, chunk):
        L = grid.levels[a:a + chunk, None]
        w = np.exp(-0.5 * ((L - x0[None, :]) / h) ** 2) * d[None, :]
        sw = w.sum(axis=1)
        sw2 = (w ** 2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m1 = (w * r[None, :]).sum(axis=1) / sw
            m2 = (w * r[None, :] ** 2).sum(axis=1) / sw
            var = np.maximum(m2 - m1 ** 2, 0.0)
            ne = np.where(sw2 > 0, sw ** 2 / sw2, 0.0)
        mu[a:a + chunk] = m1
        sig[a:a + chunk] = np.sqrt(var)
        neff[a:a + chunk] = ne
    return pd.DataFrame({"level": grid.levels, "mu": mu, "sigma": sig, "n_eff": neff}).set_index("level")
