"""Band-reversion strategy on top of rolling forecasts.

Signal s_t is either the log close ("price") or the field's centre of gravity ("gcenter",
g-weighted mean of levels with g > 0). The band is the h-step forecast that *targets* date t
(issued h sessions earlier), from the field or from Brownian motion.

    z_t = (s_t - q50_t) / (q90_t - q10_t) * 2.563      (~ standard normal units)

Entry: z_t < -z_in            -> open one unit (long) at next session's open... we use close-to-close.
Exit : position is 'armed' once z_t > +z_out; sells at the first t after arming with |z_t| < z_norm.
       Optional max_hold sessions as a safety exit. Unlimited overlapping units, fixed size.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from .forecast import RollingResult

Z80 = 2.5631  # q90 - q10 of a standard normal


@dataclass
class StrategyParams:
    signal: str = "price"      # price | gcenter
    band: str = "field"        # field | brownian
    h: int = 5
    z_in: float = 1.5
    z_out: float = 1.0
    z_norm: float = 0.5
    max_hold: int = 60
    cost_bp: float = 1.0
    sizing: str = "fixed"      # fixed | accel | decel : size of the k-th stacked unit = ratio**k (accel) or ratio**-k (decel)
    ratio: float = 1.0
    max_stack: int = 4         # exponent cap for accel/decel sizing
    side: int = 1              # +1 long (buy below band), -1 short mirror (sell above band)
    max_units: float = float("inf")  # cap on concurrent open notional; entries that would exceed it are skipped
    capitulate: float = float("inf") # flatten the whole stack when its unrealised loss (units) reaches this
    # z_norm = None  -> sell immediately when armed (single-step exit)


def signal_frame(rr: RollingResult, bars_log_close: np.ndarray, h: int, band: str, signal: str) -> pd.DataFrame:
    """One row per target date t with s_t, q10/q50/q90 (log), z_t, close."""
    fc = rr.forecasts[rr.forecasts.h == h].copy()
    pre = "m_" if band == "field" else "b_"
    fc["target"] = fc["t"] - 1 + h                      # bar index the forecast is about
    fc = fc.set_index("target")
    idx = fc.index.to_numpy()
    q10, q50, q90 = fc[pre + "q10"].to_numpy(), fc[pre + "q50"].to_numpy(), fc[pre + "q90"].to_numpy()
    if band == "brownian":                               # brownian q50 = x0 (no drift)
        q50 = fc["x0"].to_numpy()
    x = bars_log_close[idx]
    if signal == "price":
        s = x
    else:
        # centre of gravity of the field fitted with data up to and including bar `target`
        ts = list(range(rr.forecasts["t"].min(), rr.forecasts["t"].max() + 1))
        pos = {t: i for i, t in enumerate(ts)}
        s = np.full(len(idx), np.nan)
        for k, tb in enumerate(idx):
            i = pos.get(tb + 1)
            if i is None:
                continue
            g = rr.heat_g[i]
            w = np.where(np.isfinite(g) & (g > 0), g, 0.0)
            if w.sum() > 0:
                s[k] = float((w * rr.heat_levels).sum() / w.sum())
    z = (s - q50) / np.maximum(q90 - q10, 1e-9) * Z80
    return pd.DataFrame({"bar": idx, "x": x, "s": s, "q10": q10, "q50": q50, "q90": q90, "z": z}).dropna()


def run(sf: pd.DataFrame, p: StrategyParams, close_at_end: bool = True, init_open: list[dict] | None = None) -> dict:
    """Simulate; returns trades, daily P&L in notional units (sum over open units), capital series, stats.

    Sizing: the k-th unit stacked while no sell has fired since the last flat (k = 0, 1, 2, ...)
    has size ratio**k ("accel"), ratio**-k ("decel"), or 1 ("fixed").

    init_open: positions carried in from a previous segment: [{"i": negative bar offset, "x0": entry log-price
    (already side-signed), "size", "armed"}]. Their trades report the negative index; callers map it back.
    """
    x = sf["x"].to_numpy() * p.side          # short mirror: work in negated log-price so the long logic applies
    z = sf["z"].to_numpy() * p.side
    n = len(x)
    cost = p.cost_bp / 1e4
    znorm = np.inf if p.z_norm is None else p.z_norm
    open_pos: list[dict] = [dict(q) for q in (init_open or [])]
    trades = []
    pnl = np.zeros(n)                                    # daily mark-to-market P&L in notional units
    capital = np.zeros(n)                                # notional open at each close
    stack = len(open_pos)                                # units added since the last sell
    n_capit = 0
    for i in range(n):
        r = x[i] - x[i - 1] if i > 0 else 0.0
        pnl[i] += r * sum(q["size"] for q in open_pos)
        keep = []
        sold = False
        for q in open_pos:
            if z[i] > p.z_out:
                q["armed"] = True
            held = i - q["i"]
            armed_exit = q["armed"] and abs(z[i]) < znorm
            if armed_exit or held >= p.max_hold:
                ret = x[i] - q["x0"] - 2 * cost
                trades.append({"entry": q["i"], "exit": i, "held": held, "ret": ret, "size": q["size"],
                               "pnl": ret * q["size"], "armed_exit": armed_exit, "capitulated": False})
                pnl[i] -= 2 * cost * q["size"]
                sold = True
            else:
                keep.append(q)
        open_pos = keep
        # capitulation: unrealised loss of the open stack
        if open_pos:
            unreal = sum(q["size"] * (x[i] - q["x0"]) for q in open_pos)
            if unreal <= -p.capitulate:
                for q in open_pos:
                    ret = x[i] - q["x0"] - 2 * cost
                    trades.append({"entry": q["i"], "exit": i, "held": i - q["i"], "ret": ret, "size": q["size"],
                                   "pnl": ret * q["size"], "armed_exit": False, "capitulated": True})
                    pnl[i] -= 2 * cost * q["size"]
                open_pos = []
                sold = True
                n_capit += 1
        if sold:
            stack = 0
        if z[i] < -p.z_in:
            k = min(stack, p.max_stack)
            size = {"accel": p.ratio ** k, "decel": p.ratio ** (-k)}.get(p.sizing, 1.0)
            if sum(q["size"] for q in open_pos) + size <= p.max_units + 1e-9:
                open_pos.append({"i": i, "armed": False, "size": size, "x0": float(x[i])})
                stack += 1
        capital[i] = sum(q["size"] for q in open_pos)
    open_units = [dict(q) for q in open_pos]
    if close_at_end:
        for q in open_pos:
            ret = x[n - 1] - q["x0"] - 2 * cost
            trades.append({"entry": q["i"], "exit": n - 1, "held": n - 1 - q["i"], "ret": ret, "size": q["size"],
                           "pnl": ret * q["size"], "armed_exit": False, "capitulated": False})
            pnl[n - 1] -= 2 * cost * q["size"]
    tr = pd.DataFrame(trades, columns=["entry", "exit", "held", "ret", "size", "pnl", "armed_exit", "capitulated"])
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if n else 0.0
    peak = float(capital.max()) if n else 0.0
    avg_cap = float(capital.mean()) if n else 0.0
    total = float(pnl.sum())
    stats = {"trades": len(tr), "total_ret": total, "hit": float((tr["ret"] > 0).mean()) if len(tr) else np.nan,
             "avg_ret": float(tr["ret"].mean()) if len(tr) else np.nan, "avg_held": float(tr["held"].mean()) if len(tr) else np.nan,
             "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else np.nan,
             "max_dd": dd, "max_units": int(round(peak)), "peak_capital": peak, "avg_capital": avg_cap,
             "ret_peak": total / peak if peak > 0 else np.nan, "ret_daywt": total / avg_cap if avg_cap > 0 else np.nan,
             "time_in_market": float((capital > 0).mean()), "armed_exits": float(tr["armed_exit"].mean()) if len(tr) else np.nan,
             "bh_ret": float(x[-1] - x[0]), "days": n, "capitulations": n_capit}
    return {"trades": tr, "pnl": pnl, "equity": eq, "capital": capital, "stats": stats, "open": open_units}


def grid(rrs: dict[str, RollingResult], closes: dict[str, np.ndarray], space: dict) -> pd.DataFrame:
    """Run every combination in `space` on every symbol; returns one row per (config, symbol)."""
    keys = list(space)
    rows = []
    cache = {}
    for combo in product(*space.values()):
        p = StrategyParams(**dict(zip(keys, combo)))
        for sym, rr in rrs.items():
            ck = (sym, p.h, p.band, p.signal)
            if ck not in cache:
                cache[ck] = signal_frame(rr, closes[sym], p.h, p.band, p.signal)
            st = run(cache[ck], p)["stats"]
            rows.append({**p.__dict__, "symbol": sym, **st})
    return pd.DataFrame(rows)
