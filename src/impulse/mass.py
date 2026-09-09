"""Mass m(x): decayed volume-at-price density on the grid."""
from __future__ import annotations

import numpy as np

from .bars import Bars
from .grid import PriceGrid, decay_weights


def gaussian_smooth(y: np.ndarray, dx: float, bandwidth: float) -> np.ndarray:
    """Gaussian kernel smoothing on an even grid (renormalised at the edges)."""
    if bandwidth <= 0:
        return y.copy()
    half = int(np.ceil(4 * bandwidth / dx))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * dx / bandwidth) ** 2)
    num = np.convolve(y, k, mode="same")
    den = np.convolve(np.ones_like(y), k, mode="same")
    return num / den


def volume_at_price(bars: Bars, grid: PriceGrid, tau: float | None = None) -> np.ndarray:
    """Spread each bar's (decay-weighted) volume uniformly over [log low, log high].

    Returns raw mass per bin (not smoothed). Bar ranges narrower than a bin still
    deposit their full volume into the bin(s) they overlap.
    """
    lo = bars.log_low
    hi = np.maximum(bars.log_high, lo + 1e-12)
    w = bars.volume * decay_weights(len(bars), tau)
    edges = grid.edges
    m = np.zeros(grid.n)
    # overlap of [lo,hi] with every bin: vectorised via cumulative coverage
    # coverage(e) = clip((e - lo)/(hi - lo), 0, 1); mass in bin j = w*(cov(e_{j+1}) - cov(e_j))
    cov = np.clip((edges[None, :] - lo[:, None]) / (hi - lo)[:, None], 0.0, 1.0)
    frac = np.diff(cov, axis=1)  # (n_bars, n_bins)
    m = (w[:, None] * frac).sum(axis=0)
    return m


def mass(bars: Bars, grid: PriceGrid, tau: float | None = None, smooth: bool = True) -> np.ndarray:
    """m(x): volume-at-price density, kernel-smoothed with the grid bandwidth, normalised to sum 1."""
    m = volume_at_price(bars, grid, tau)
    if smooth:
        m = gaussian_smooth(m, grid.dx, grid.bandwidth)
    s = m.sum()
    return m / s if s > 0 else m
