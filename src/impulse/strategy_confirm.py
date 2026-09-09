"""Confirmation-and-acceleration short.

State machine on the band z-score (z = (x - q50)/(q90 - q10) * 2.563):

  SETUP   : a close with z > z_break within the last `lookback` sessions ("broke bounds")
  ENTRY   : first close after a setup with z < z_enter (back inside, lower half) -> open `start` units
  ADD     : while short, each close whose z is at least `step` below the z at the last entry -> add prev * accel
            (accel < 1 = decelerating stack: 0.25, 0.125, 0.0625 ... for accel 0.5)
  EXIT    : any close with z > z_exit (back to the top part of the range) -> cover everything
            also: max_hold sessions, or unrealised loss >= capitulate units
Sizes are notional units (1 = one full-size unit). Long mirror not implemented: this is a short-only rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConfirmParams:
    z_break: float = 1.0
    z_enter: float = 0.0
    step: float = 0.5
    z_exit: float = 0.5
    start: float = 0.25
    accel: float = 1.5
    max_adds: int = 5
    lookback: int = 20
    max_hold: int = 60
    capitulate: float = float("inf")
    cost_bp: float = 1.0
    rearm: bool = False        # keep the setup alive after an entry so a later dip re-enters within the lookback
    side: int = -1             # short-only rule (kept for interface parity with StrategyParams)


def run_confirm(sf: pd.DataFrame, p: ConfirmParams, close_at_end: bool = True, init_open: list[dict] | None = None) -> dict:
    """init_open: carried positions [{i (negative offset), x0 (raw log price), size, z0}] from a previous segment."""
    x = sf["x"].to_numpy(); z = sf["z"].to_numpy(); n = len(x)
    cost = p.cost_bp / 1e4
    pos: list[dict] = [dict(q) for q in (init_open or [])]          # each: i, x0, size, z0
    trades = []
    pnl = np.zeros(n); capital = np.zeros(n)
    last_break = -10**9
    last_entry_z = pos[-1].get("z0") if pos else None
    n_capit = 0
    for i in range(n):
        r = x[i] - x[i - 1] if i > 0 else 0.0
        units = sum(q["size"] for q in pos)
        pnl[i] -= r * units                              # short: gain when price falls
        if z[i] > p.z_break:
            last_break = i
        if pos:
            unreal = -sum(q["size"] * (x[i] - q["x0"]) for q in pos)
            held = i - pos[0]["i"]
            exit_top = z[i] > p.z_exit
            timeout = held >= p.max_hold
            capit = unreal <= -p.capitulate
            if exit_top or timeout or capit:
                for q in pos:
                    ret = -(x[i] - q["x0"]) - 2 * cost
                    trades.append({"entry": q["i"], "exit": i, "held": i - q["i"], "ret": ret, "size": q["size"], "pnl": ret * q["size"],
                                   "reason": "capitulate" if capit else ("timeout" if timeout else "top"),
                                   "armed_exit": bool(exit_top and not capit and not timeout), "capitulated": bool(capit)})
                    pnl[i] -= 2 * cost * q["size"]
                if capit:
                    n_capit += 1
                pos = []; last_entry_z = None
            elif len(pos) - 1 < p.max_adds and last_entry_z is not None and z[i] <= last_entry_z - p.step:
                size = pos[-1]["size"] * p.accel
                pos.append({"i": i, "x0": float(x[i]), "size": size, "z0": float(z[i])}); last_entry_z = z[i]
        else:
            if i - last_break <= p.lookback and last_break >= 0 and z[i] < p.z_enter and z[i] <= p.z_exit:
                pos.append({"i": i, "x0": float(x[i]), "size": p.start, "z0": float(z[i])}); last_entry_z = z[i]
                if not p.rearm:
                    last_break = -10**9                   # one entry per setup
        capital[i] = sum(q["size"] for q in pos)
    open_units = [{"i": q["i"], "x0": q["x0"], "size": q["size"], "z0": q.get("z0"), "armed": False} for q in pos]
    if close_at_end:
        for q in pos:
            ret = -(x[n - 1] - q["x0"]) - 2 * cost
            trades.append({"entry": q["i"], "exit": n - 1, "held": n - 1 - q["i"], "ret": ret, "size": q["size"], "pnl": ret * q["size"], "reason": "end",
                           "armed_exit": False, "capitulated": False})
            pnl[n - 1] -= 2 * cost * q["size"]
    tr = pd.DataFrame(trades, columns=["entry", "exit", "held", "ret", "size", "pnl", "reason", "armed_exit", "capitulated"])
    eq = np.cumsum(pnl); dd = float((np.maximum.accumulate(eq) - eq).max()) if n else 0.0
    peak = float(capital.max()) if n else 0.0; avg = float(capital.mean()) if n else 0.0; total = float(pnl.sum())
    stats = {"trades": len(tr), "total_ret": total, "hit": float((tr["ret"] > 0).mean()) if len(tr) else np.nan,
             "avg_held": float(tr["held"].mean()) if len(tr) else np.nan, "peak_capital": peak, "avg_capital": avg,
             "ret_peak": total / peak if peak > 0 else np.nan, "ret_daywt": total / avg if avg > 0 else np.nan, "max_dd": dd,
             "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else np.nan,
             "bh_ret": float(-(x[-1] - x[0])), "capitulations": n_capit, "time_in_market": float((capital > 0).mean()),
             "stacks": int((tr["reason"] != "").sum()) if len(tr) else 0, "armed_exits": float(tr["armed_exit"].mean()) if len(tr) else np.nan}
    return {"trades": tr, "pnl": pnl, "equity": eq, "capital": capital, "stats": stats, "open": open_units}
