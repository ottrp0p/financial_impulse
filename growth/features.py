"""Open->close decomposition and features known at the open.

For session t the field is fitted on bars[t-fit, t) (closes through t-1); everything below uses only
that window plus open_t. Targets (oc, lo_oc, hi_oc) are stored alongside but never fed to a feature.
All z-quantities are in units of the window's per-bar close-to-close sigma (sigma_bar), so they pool
across names.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import ROOT  # noqa: F401
from impulse.bars import Bars
from impulse.grid import PriceGrid
from impulse.potential import fit_field

FIELD_FEATURES = ["mu_z", "g_open", "z_mass", "z_restoring", "z_calm", "sigma_ratio", "peak_dist"]
BASE_FEATURES = ["gap_z", "ret1_z", "ret5_z", "oc_prev_z", "vol_ratio"]
FEATURE_SETS = {"field": FIELD_FEATURES, "base": BASE_FEATURES, "both": FIELD_FEATURES + BASE_FEATURES}


def _interp(levels: np.ndarray, v: np.ndarray, x: float, fill: float = 0.0) -> float:
    ok = np.isfinite(v)
    if ok.sum() < 2 or x < levels[ok][0] or x > levels[ok][-1]:
        return fill
    return float(np.interp(x, levels[ok], v[ok]))


def _peak_dist(levels: np.ndarray, m: np.ndarray, valid: np.ndarray, x: float, s: float) -> float:
    """Signed distance (sigma units) from x to the nearest local maximum of mass among valid levels."""
    mm = np.where(valid, m, -np.inf)
    pk = [i for i in range(1, len(mm) - 1) if mm[i] > mm[i - 1] and mm[i] >= mm[i + 1] and np.isfinite(mm[i])]
    if not pk:
        return 0.0
    d = (levels[pk] - x) / s
    return float(np.clip(d[np.argmin(np.abs(d))], -5, 5))


def symbol_features(bars: Bars, fit_bars: int = 120, shrink_n0: float = 50.0, min_n_eff: float = 10.0,
                    step: int = 1) -> pd.DataFrame:
    df = bars.df
    x = np.log(df["close"].to_numpy())
    o = np.log(df["open"].to_numpy())
    lo = np.log(df["low"].to_numpy())
    hi = np.log(df["high"].to_numpy())
    adv = (df["close"] * df["volume"]).rolling(20, min_periods=10).median().shift(1).to_numpy()
    n = len(df)
    rows = []
    fld = None
    for t in range(fit_bars, n):
        if fld is None or (t - fit_bars) % step == 0:
            fit = bars.slice(t - fit_bars, t)
            try:
                grid = PriceGrid.from_bars(fit)
                fld = fit_field(fit, grid, tau=fit_bars / 2, min_n_eff=min_n_eff)
            except ValueError:  # rare grid/mass length mismatch inside fit_field; reuse last field
                if fld is None:
                    continue
            else:
                tb = fld.table
                levels = grid.levels
                valid = tb["valid"].to_numpy()
                ne = tb["n_eff"].to_numpy()
                mu = np.where(valid, tb["mu"].to_numpy() * ne / (ne + shrink_n0), np.nan)
                sg = np.where(valid, tb["sigma"].to_numpy(), np.nan)
                g = tb["g"].to_numpy()
                zm, zr, zc = tb["z_mass"].to_numpy(), tb["z_restoring"].to_numpy(), tb["z_calm"].to_numpy()
                mass = tb["mass"].to_numpy()
        r = np.diff(x[t - fit_bars:t])
        s = float(np.std(r))
        if not np.isfinite(s) or s <= 0:
            continue
        xo = o[t]
        rows.append({
            "ts": df.index[t], "symbol": bars.symbol,
            # --- field at the open ---
            "mu_z": _interp(levels, mu, xo) / s,
            "g_open": _interp(levels, g, xo),
            "z_mass": _interp(levels, zm, xo), "z_restoring": _interp(levels, zr, xo), "z_calm": _interp(levels, zc, xo),
            "sigma_ratio": _interp(levels, sg, xo, fill=s) / s,
            "peak_dist": _peak_dist(levels, mass, valid, xo, s),
            # --- baseline ---
            "gap_z": (xo - x[t - 1]) / s,
            "ret1_z": (x[t - 1] - x[t - 2]) / s,
            "ret5_z": (x[t - 1] - x[t - 6]) / (s * np.sqrt(5)),
            "oc_prev_z": (x[t - 1] - o[t - 1]) / s,
            "vol_ratio": float(np.std(r[-5:])) / s,
            # --- state ---
            "sigma_bar": s, "adv": adv[t], "open": float(df["open"].iloc[t]),
            # --- targets (never features) ---
            "oc": x[t] - xo, "lo_oc": lo[t] - xo, "hi_oc": hi[t] - xo,
        })
    out = pd.DataFrame(rows)
    out["oc_z"] = out["oc"] / out["sigma_bar"]
    return out


def _worker(args):
    bars, kw = args
    return symbol_features(bars, **kw)


def panel_features(bars_by_sym: dict[str, Bars], workers: int = 1, **kw) -> pd.DataFrame:
    items = [(b, kw) for b in bars_by_sym.values()]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_worker, items))
    else:
        parts = [_worker(i) for i in items]
    p = pd.concat([q for q in parts if len(q)], ignore_index=True)
    return p.sort_values(["ts", "symbol"]).reset_index(drop=True)
