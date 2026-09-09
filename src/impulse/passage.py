"""Passage statistics: how price behaves while inside a band around each level.

An *episode* is a maximal run of consecutive bars whose log close lies within
[level - h, level + h]. For each episode we record residence (bars), the side
price entered from, whether it left on the same side (reversal) or the other
side (traversal), and the mean absolute one-bar log return while inside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bars import Bars
from .grid import PriceGrid

EPISODE_COLUMNS = ["level_idx", "level", "start", "end", "residence", "entered_from",
                   "exited_to", "reversal", "abs_ret", "net_move"]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def episodes(bars: Bars, grid: PriceGrid, half_width: float | None = None,
             min_residence: int = 1) -> pd.DataFrame:
    """All band episodes for every grid level. Episodes touching the series
    boundary (no observed entry or exit) are dropped: their side is unknown."""
    h = grid.bandwidth if half_width is None else half_width
    x = bars.log_close
    r = np.abs(np.diff(x, prepend=x[0]))  # |return| attributed to the bar it lands on
    n = len(x)
    rows: list[tuple] = []
    for k, L in enumerate(grid.levels):
        inside = np.abs(x - L) <= h
        for s, e in _runs(inside):
            if s == 0 or e == n:
                continue
            res = e - s
            if res < min_residence:
                continue
            entered_from = -1 if x[s - 1] < L else 1  # -1: from below, +1: from above
            exited_to = -1 if x[e] < L else 1
            rows.append((k, L, s, e, res, entered_from, exited_to, entered_from == exited_to,
                         float(r[s:e].mean()) if e > s else 0.0, float(x[e - 1] - x[s])))
    return pd.DataFrame(rows, columns=EPISODE_COLUMNS)


def level_stats(eps: pd.DataFrame, grid: PriceGrid) -> pd.DataFrame:
    """Aggregate episodes per level: n, median/mean residence, reversal prob, mean |ret|."""
    out = pd.DataFrame(index=pd.Index(grid.levels, name="level"))
    if len(eps) == 0:
        for c in ("n_episodes", "residence_median", "residence_mean", "reversal_prob", "abs_ret_mean"):
            out[c] = np.nan
        out["n_episodes"] = 0
        return out
    g = eps.groupby("level")
    out["n_episodes"] = g.size().reindex(out.index).fillna(0).astype(int)
    out["residence_median"] = g["residence"].median().reindex(out.index)
    out["residence_mean"] = g["residence"].mean().reindex(out.index)
    out["reversal_prob"] = g["reversal"].mean().reindex(out.index)
    out["abs_ret_mean"] = g["abs_ret"].mean().reindex(out.index)
    return out
