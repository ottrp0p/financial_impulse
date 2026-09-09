"""Log-price grid, bandwidth rule, and time-decay weights shared by all estimators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bars import Bars


def silverman_bandwidth(x: np.ndarray) -> float:
    """Silverman rule of thumb on log-price. Fixed rule => no tuning on test data."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need >= 2 points for a bandwidth")
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    a = min(sd, iqr / 1.349) if iqr > 0 else sd
    if a <= 0:
        a = 1e-6
    return 1.06 * a * n ** (-0.2)


def decay_weights(n: int, tau: float | None) -> np.ndarray:
    """exp(-(T - t)/tau) for t = 0..n-1; tau in bars. tau=None => uniform."""
    if tau is None:
        return np.ones(n)
    t = np.arange(n)
    return np.exp(-(n - 1 - t) / float(tau))


@dataclass(frozen=True)
class PriceGrid:
    """Evenly spaced levels in log-price.

    `levels` are bin centres; `edges` has len(levels)+1 entries.
    `bandwidth` is the kernel width used by every level-conditioned estimator and
    the half-width of passage bands, so one number controls resolution everywhere.
    """

    levels: np.ndarray
    bandwidth: float

    @property
    def dx(self) -> float:
        return float(self.levels[1] - self.levels[0]) if len(self.levels) > 1 else self.bandwidth

    @property
    def edges(self) -> np.ndarray:
        return np.concatenate([self.levels - self.dx / 2, [self.levels[-1] + self.dx / 2]])

    @property
    def n(self) -> int:
        return len(self.levels)

    @classmethod
    def from_range(cls, lo: float, hi: float, bandwidth: float, bins_per_bandwidth: int = 2,
                   pad_bandwidths: float = 1.0) -> "PriceGrid":
        dx = bandwidth / bins_per_bandwidth
        lo -= pad_bandwidths * bandwidth
        hi += pad_bandwidths * bandwidth
        n = max(int(np.ceil((hi - lo) / dx)) + 1, 3)
        levels = lo + dx * np.arange(n)
        return cls(levels=levels, bandwidth=float(bandwidth))

    @classmethod
    def from_bars(cls, bars: Bars, bandwidth: float | None = None, **kw) -> "PriceGrid":
        bw = silverman_bandwidth(bars.log_close) if bandwidth is None else bandwidth
        return cls.from_range(bars.log_low.min(), bars.log_high.max(), bw, **kw)

    def index_of(self, x: np.ndarray) -> np.ndarray:
        """Nearest level index for each log-price (clipped to the grid)."""
        i = np.rint((np.asarray(x) - self.levels[0]) / self.dx).astype(int)
        return np.clip(i, 0, self.n - 1)
