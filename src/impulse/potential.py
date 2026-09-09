"""Potential U(x), stationary density check, and the gravity score g(x).

The gravity score is a *hypothesis*, not a fit: an explicit weighted mean of
z-scored components. Weights are configurable and reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bars import Bars
from .dynamics import local_dynamics
from .grid import PriceGrid
from .mass import mass
from .passage import episodes, level_stats

DEFAULT_WEIGHTS = {"mass": 0.5, "calm": 1.0, "restoring": 1.0, "residence": 1.0}


def potential(mu: np.ndarray, dx: float) -> np.ndarray:
    """U(x) = -integral mu dx (cumulative trapezoid), anchored so min U = 0."""
    mu = np.nan_to_num(mu, nan=0.0)
    U = -np.concatenate([[0.0], np.cumsum(0.5 * (mu[1:] + mu[:-1]) * dx)])
    return U - U.min()


def stationary_density(U: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """p(x) ~ exp(-2 U / sigma^2) for the Fokker-Planck stationary solution
    (constant-sigma approximation; used only as a self-consistency check vs m(x))."""
    s2 = np.nanmedian(sigma) ** 2 if np.all(np.isnan(sigma)) is False else 1.0
    p = np.exp(-2.0 * (U - U.min()) / s2)
    return p / p.sum()


def zscore(v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full_like(v, np.nan, dtype=float)
    x = v[valid]
    sd = np.nanstd(x)
    if sd > 0:
        out[valid] = (x - np.nanmean(x)) / sd
    else:
        out[valid] = 0.0
    return out


@dataclass
class Field:
    """Everything estimated for one fit window, indexed by grid level."""

    grid: PriceGrid
    table: pd.DataFrame  # columns: mass, mu, sigma, n_eff, U, p_stat, residence_median, n_episodes, z_*, g, valid
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    tau: float | None = None
    band_half_width: float = 0.0  # passage band used for residence (>= grid.bandwidth)

    @property
    def g(self) -> np.ndarray:
        return self.table["g"].to_numpy()

    @property
    def valid(self) -> np.ndarray:
        return self.table["valid"].to_numpy()


def band_half_width(bars: Bars, grid: PriceGrid, band_sigmas: float = 3.0) -> float:
    """Half-width of passage bands: max(kernel bandwidth, band_sigmas * per-bar return sd).

    Residence inside a band narrower than ~one bar's move is set by diffusion alone
    (every episode lasts ~1 bar), so drift is invisible; bands must be a few
    per-bar sigmas wide for stickiness to be measurable.
    """
    s_bar = float(np.std(np.diff(bars.log_close))) if len(bars) > 2 else 0.0
    return max(grid.bandwidth, band_sigmas * s_bar)


def fit_field(bars: Bars, grid: PriceGrid | None = None, tau: float | None = None,
              weights: dict | None = None, min_n_eff: float = 20.0,
              min_episodes: int = 3, band_sigmas: float = 3.0) -> Field:
    """Estimate m, mu, sigma, U, residence and combine into g(x) on `grid`."""
    grid = PriceGrid.from_bars(bars) if grid is None else grid
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    h_pass = band_half_width(bars, grid, band_sigmas)

    m = mass(bars, grid, tau)
    dyn = local_dynamics(bars, grid, tau)
    U = potential(dyn["mu"].to_numpy(), grid.dx)
    eps = episodes(bars, grid, half_width=h_pass)
    ls = level_stats(eps, grid)

    t = pd.DataFrame(index=pd.Index(grid.levels, name="level"))
    t["mass"] = m
    t["mu"] = dyn["mu"].to_numpy()
    t["sigma"] = dyn["sigma"].to_numpy()
    t["n_eff"] = dyn["n_eff"].to_numpy()
    t["U"] = U
    t["p_stat"] = stationary_density(U, t["sigma"].to_numpy())
    t["residence_median"] = ls["residence_median"].to_numpy()
    t["n_episodes"] = ls["n_episodes"].to_numpy()

    valid = (t["n_eff"].to_numpy() >= min_n_eff) & (t["mass"].to_numpy() > 0) & (t["sigma"].to_numpy() > 0)
    t["valid"] = valid

    # components (higher = more gravity)
    comps = {
        "mass": np.log(np.where(t["mass"] > 0, t["mass"], np.nan)),
        "calm": -np.log(np.where(t["sigma"] > 0, t["sigma"], np.nan)),
        # restoring force: -d mu/dx > 0 means drift points back toward the level
        "restoring": -np.gradient(np.nan_to_num(t["mu"].to_numpy()), grid.dx),
        "residence": np.log(t["residence_median"].to_numpy()),
    }
    num = np.zeros(grid.n)
    den = np.zeros(grid.n)
    for name, v in comps.items():
        wt = w.get(name, 0.0)
        v = np.asarray(v, dtype=float)
        ok = valid & np.isfinite(v)
        if name == "residence":
            ok &= t["n_episodes"].to_numpy() >= min_episodes
        z = zscore(v, ok)
        t[f"z_{name}"] = z
        if wt == 0:
            continue
        use = np.isfinite(z)
        num[use] += wt * z[use]
        den[use] += wt
    g = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    g[~valid] = np.nan
    t["g"] = g
    return Field(grid=grid, table=t, weights=w, tau=tau, band_half_width=h_pass)
