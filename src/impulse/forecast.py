"""Probabilistic h-step forecasts from the fitted field, scored against a Brownian baseline.

Model:    dx = mu(x) dt + sigma(x) dW   (mu, sigma interpolated from the fitted Field; outside the
          valid region the process falls back to zero drift and the window's per-bar sigma)
Baseline: dx = sigma_bar dW             (pure Brownian, zero drift, trailing per-bar sigma)

Scores per forecast: log predictive density at the realised value (KDE on samples), CRPS, PIT,
and 10/50/90 quantiles. Skill = model minus baseline, so > 0 means better than chance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .bars import Bars
from .grid import PriceGrid
from .potential import Field, fit_field


def _interp_field(fld: Field, shrink_n0: float = 0.0, sigma_mode: str = "local"):
    """mu is shrunk toward 0 by n_eff/(n_eff+shrink_n0); sigma is 'local', 'bar' (constant), or 'blend'."""
    t = fld.table
    ok = t["valid"].to_numpy()
    L = fld.grid.levels[ok]
    ne = t["n_eff"].to_numpy()[ok]
    mu = t["mu"].to_numpy()[ok] * (ne / (ne + shrink_n0) if shrink_n0 > 0 else 1.0)
    sg = t["sigma"].to_numpy()[ok]
    lo, hi = (L[0], L[-1]) if len(L) else (np.nan, np.nan)

    def f(x: np.ndarray, sigma_bar: float):
        if len(L) < 2:
            return np.zeros_like(x), np.full_like(x, sigma_bar)
        inside = (x >= lo) & (x <= hi)
        m = np.where(inside, np.interp(x, L, mu), 0.0)
        if sigma_mode == "bar":
            s = np.full_like(x, sigma_bar)
        else:
            s_loc = np.where(inside, np.interp(x, L, sg), sigma_bar)
            s = 0.5 * (s_loc + sigma_bar) if sigma_mode == "blend" else s_loc
        return m, s
    return f


def model_samples(fld: Field, x0: float, h: int, sigma_bar: float, n: int = 2000,
                  rng: np.random.Generator | None = None, substeps: int = 1,
                  shrink_n0: float = 0.0, sigma_mode: str = "local") -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    f = _interp_field(fld, shrink_n0, sigma_mode)
    dt = 1.0 / substeps
    x = np.full(n, x0, dtype=float)
    for _ in range(h * substeps):
        m, s = f(x, sigma_bar)
        x = x + m * dt + s * np.sqrt(dt) * rng.standard_normal(n)
    return x


def brownian_samples(x0: float, h: int, sigma_bar: float, n: int = 2000,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return x0 + sigma_bar * np.sqrt(h) * rng.standard_normal(n)


def score_samples(samples: np.ndarray, y: float) -> dict:
    """log density (Gaussian KDE), CRPS (sample formula), PIT, quantiles."""
    s = np.sort(samples)
    kde = stats.gaussian_kde(s)
    logp = float(np.log(max(kde(y)[0], 1e-300)))
    n = len(s)
    crps = float(np.mean(np.abs(s - y)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :])) if n <= 600
                 else np.mean(np.abs(s - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / (n * n))
    pit = float(np.searchsorted(s, y) / n)
    q = np.quantile(s, [0.1, 0.5, 0.9])
    return {"logp": logp, "crps": crps, "pit": pit, "q10": float(q[0]), "q50": float(q[1]), "q90": float(q[2])}


@dataclass
class RollingResult:
    forecasts: pd.DataFrame      # one row per (t, horizon): model & baseline scores
    heat_levels: np.ndarray      # log-price grid for the heatmaps
    heat_g: np.ndarray           # (n_dates, n_levels) rolling g(x) fitted at each date (NaN outside valid)
    heat_density: np.ndarray     # (n_dates, n_levels) model predictive density for `density_h` steps ahead
    density_h: int


def rolling_forecast(bars: Bars, fit_bars: int = 120, horizons=(1, 5, 10), n_paths: int = 2000,
                     density_h: int = 5, seed: int = 0, min_n_eff: float = 10.0,
                     heat_levels: np.ndarray | None = None, shrink_n0: float = 0.0,
                     sigma_mode: str = "local") -> RollingResult:
    rng = np.random.default_rng(seed)
    x = bars.log_close
    hmax = max(horizons)
    ts = list(range(fit_bars, len(bars) - hmax + 1))
    if heat_levels is None:
        lo, hi = x.min(), x.max()
        heat_levels = np.linspace(lo - 0.02, hi + 0.02, 160)
    hg = np.full((len(ts), len(heat_levels)), np.nan)
    hd = np.zeros((len(ts), len(heat_levels)))
    rows = []
    for i, t in enumerate(ts):
        fit = bars.slice(t - fit_bars, t)
        grid = PriceGrid.from_bars(fit)
        fld = fit_field(fit, grid, tau=fit_bars / 2, min_n_eff=min_n_eff)
        sigma_bar = float(np.std(np.diff(fit.log_close)))
        x0 = x[t - 1]
        # rolling g(x) on the common heat grid
        g = fld.g
        ok = np.isfinite(g)
        if ok.sum() >= 2:
            hg[i] = np.interp(heat_levels, grid.levels[ok], g[ok], left=np.nan, right=np.nan)
            edge = (heat_levels < grid.levels[ok][0]) | (heat_levels > grid.levels[ok][-1])
            hg[i, edge] = np.nan
        for h in horizons:
            y = x[t - 1 + h]
            sm = model_samples(fld, x0, h, sigma_bar, n_paths, rng, shrink_n0=shrink_n0, sigma_mode=sigma_mode)
            sb = brownian_samples(x0, h, sigma_bar, n_paths, rng)
            m, b = score_samples(sm, y), score_samples(sb, y)
            rows.append({"t": t, "date": bars.ts[t - 1], "h": h, "x0": x0, "y": y,
                         **{f"m_{k}": v for k, v in m.items()}, **{f"b_{k}": v for k, v in b.items()}})
            if h == density_h:
                kde = stats.gaussian_kde(sm)
                hd[i] = kde(heat_levels)
    return RollingResult(pd.DataFrame(rows), heat_levels, hg, hd, density_h)


def skill_summary(fc: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per horizon: mean log-likelihood gain (nats) with block-bootstrap CI, CRPS skill,
    80% interval coverage and width ratio, PIT dispersion."""
    rng = np.random.default_rng(seed)
    out = []
    for h, d in fc.groupby("h"):
        gain = (d["m_logp"] - d["b_logp"]).to_numpy()
        n = len(gain)
        block = max(int(h), 1)
        nb = int(np.ceil(n / block))
        boots = []
        for _ in range(n_boot):
            starts = rng.integers(0, max(n - block + 1, 1), size=nb)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            boots.append(gain[idx].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        cov_m = float(((d["y"] >= d["m_q10"]) & (d["y"] <= d["m_q90"])).mean())
        cov_b = float(((d["y"] >= d["b_q10"]) & (d["y"] <= d["b_q90"])).mean())
        w_m = float((d["m_q90"] - d["m_q10"]).mean())
        w_b = float((d["b_q90"] - d["b_q10"]).mean())
        out.append({"h": h, "n": n, "loglik_gain": float(gain.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                    "frac_better": float((gain > 0).mean()),
                    "crps_skill": float(1 - d["m_crps"].mean() / d["b_crps"].mean()),
                    "cov80_model": cov_m, "cov80_brownian": cov_b, "width_ratio": w_m / w_b,
                    "pit_sd_model": float(d["m_pit"].std()), "pit_sd_brownian": float(d["b_pit"].std())})
    return pd.DataFrame(out)
