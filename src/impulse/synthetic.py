"""Synthetic price processes with known potentials, rendered as OHLCV bars.

Each generator simulates log-price on a fine sub-step grid and aggregates
`substeps` steps into one bar so high/low are meaningful.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .bars import Bars


def _to_bars(path: np.ndarray, substeps: int, volume: np.ndarray | None, rng: np.random.Generator,
             symbol: str, start: str = "2020-01-01", freq: str = "1h") -> Bars:
    n_bars = (len(path) - 1) // substeps
    seg = path[: n_bars * substeps + 1]
    # bar i covers seg[i*s : i*s + s + 1]; open = first, close = last
    blocks = np.lib.stride_tricks.sliding_window_view(seg, substeps + 1)[::substeps][:n_bars]
    o, c = blocks[:, 0], blocks[:, -1]
    h, l = blocks.max(axis=1), blocks.min(axis=1)
    if volume is None:
        volume = rng.lognormal(mean=10.0, sigma=0.5, size=n_bars)
    idx = pd.date_range(start, periods=n_bars, freq=freq, tz="UTC")
    df = pd.DataFrame({"open": np.exp(o), "high": np.exp(h), "low": np.exp(l), "close": np.exp(c),
                       "volume": volume}, index=idx)
    return Bars(df, symbol=symbol, interval=freq)


def simulate_langevin(force: Callable[[np.ndarray], np.ndarray], x0: float, n_bars: int,
                      sigma: float, substeps: int = 10, seed: int = 0, symbol: str = "SYN") -> Bars:
    """dx = force(x) dt + sigma dW in per-bar units (dt = 1 bar), Euler-Maruyama."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / substeps
    n = n_bars * substeps
    x = np.empty(n + 1)
    x[0] = x0
    noise = rng.normal(0.0, sigma * np.sqrt(dt), size=n)
    for i in range(n):
        x[i + 1] = x[i] + force(x[i]) * dt + noise[i]
    return _to_bars(x, substeps, None, rng, symbol)


def ou(n_bars: int = 5000, k: float = 0.05, s_star: float = 0.0, sigma: float = 0.02,
       seed: int = 0, **kw) -> Bars:
    """Ornstein-Uhlenbeck: single well at s_star, mu(x) = -k (x - s_star)."""
    return simulate_langevin(lambda x: -k * (x - s_star), s_star, n_bars, sigma, seed=seed,
                             symbol="OU", **kw)


def double_well(n_bars: int = 20000, a: float = 2e-4, width: float = 0.1, sigma: float = 0.01,
                seed: int = 0, **kw) -> Bars:
    """U(x) = a ((x/width)^2 - 1)^2 : wells at +-width, barrier at 0. force = -U'."""
    def force(x):
        u = x / width
        return -4.0 * a * u * (u * u - 1.0) / width
    return simulate_langevin(force, width, n_bars, sigma, seed=seed, symbol="DW", **kw)


def random_walk(n_bars: int = 5000, sigma: float = 0.02, seed: int = 0, **kw) -> Bars:
    """Driftless random walk: flat potential (null model)."""
    return simulate_langevin(lambda x: 0.0 * x, 0.0, n_bars, sigma, seed=seed, symbol="RW", **kw)


def random_walk_with_volume_spike(n_bars: int = 5000, sigma: float = 0.02, x0: float = 0.0,
                                  half_width: float = 0.01, factor: float = 20.0,
                                  seed: int = 0, **kw) -> Bars:
    """Random walk whose volume is `factor`x larger whenever close is within
    half_width of x0. Mass concentrates at x0 but the dynamics are unchanged."""
    bars = random_walk(n_bars, sigma, seed, **kw)
    v = bars.volume.copy()
    near = np.abs(bars.log_close - x0) <= half_width
    v[near] *= factor
    df = bars.df.copy()
    df["volume"] = v
    return Bars(df, symbol="RWV", interval=bars.interval)
