"""Ledgers: backtest (always running) and live (flat at inception, no forced close today), plus today's actions.

Both are the same stack machine (impulse.strategy.run) over the same signal frame; they differ only in
where the frame starts and whether the last bar is force-closed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from impulse.strategy import Z80, StrategyParams, run as run_stack
from impulse.strategy_confirm import ConfirmParams, run_confirm

ANN = np.sqrt(252)


def run(sf, p, close_at_end=True, init_open=None):
    """Dispatch on the params type."""
    if isinstance(p, ConfirmParams):
        return run_confirm(sf, p, close_at_end=close_at_end, init_open=init_open)
    return run_stack(sf, p, close_at_end=close_at_end, init_open=init_open)


def to_params(cfg: dict):
    if cfg.get("kind") == "confirm":
        return ConfirmParams(z_break=float(cfg["z_break"]), z_enter=float(cfg["z_enter"]), step=float(cfg["step"]), z_exit=float(cfg["z_exit"]),
                             start=float(cfg["start"]), accel=float(cfg["accel"]), max_adds=int(cfg["max_adds"]), lookback=int(cfg["lookback"]),
                             max_hold=int(cfg["max_hold"]), capitulate=float("inf") if cfg.get("capitulate") is None else float(cfg["capitulate"]),
                             cost_bp=float(cfg.get("cost_bp", 1.0)), rearm=bool(cfg.get("rearm", False)), side=-1)
    return StrategyParams(signal="price", band=cfg["band"], h=int(cfg["h"]), z_in=float(cfg["z_in"]), z_out=float(cfg["z_out"]),
                          z_norm=None if cfg.get("z_norm") is None else float(cfg["z_norm"]), max_hold=int(cfg["max_hold"]),
                          cost_bp=float(cfg.get("cost_bp", 1.0)), sizing=cfg["sizing"], ratio=float(cfg["ratio"]), max_stack=int(cfg.get("max_stack", 4)),
                          side=int(cfg["side"]), max_units=float("inf") if cfg.get("max_units") is None else float(cfg["max_units"]),
                          capitulate=float("inf") if cfg.get("capitulate") is None else float(cfg["capitulate"]))


def signal_frame(prices: pd.DataFrame, fc: pd.DataFrame, band: str) -> pd.DataFrame:
    """One row per date that has a band targeting it: date, x, q10/q50/q90 (log), z."""
    dates = [d.strftime("%Y-%m-%d") for d in prices.index]
    pos = {d: i for i, d in enumerate(dates)}
    x = np.log(prices["close"].to_numpy())
    pre = "m_" if band == "field" else "b_"
    rows = []
    for r in fc.itertuples():
        i = pos.get(r.issue_date)
        if i is None:
            continue
        tgt = i + int(r.h)
        if tgt >= len(dates):
            continue
        q10, q50, q90 = getattr(r, pre + "q10"), getattr(r, pre + "q50"), getattr(r, pre + "q90")
        rows.append((dates[tgt], x[tgt], q10, q50, q90))
    sf = pd.DataFrame(rows, columns=["date", "x", "q10", "q50", "q90"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    sf["z"] = (sf["x"] - sf["q50"]) / np.maximum(sf["q90"] - sf["q10"], 1e-9) * Z80
    return sf


def pending_band(prices: pd.DataFrame, fc: pd.DataFrame, band: str) -> dict | None:
    """The band that will target the *next* session (for display)."""
    return None


def run_window(sf: pd.DataFrame, p: StrategyParams, start: str | None, end: str | None, close_at_end: bool) -> dict:
    m = np.ones(len(sf), dtype=bool)
    if start:
        m &= sf["date"].to_numpy() >= start
    if end:
        m &= sf["date"].to_numpy() <= end
    w = sf[m].reset_index(drop=True)
    if len(w) < 1:
        return {"frame": w, "res": None}
    res = run(w, p, close_at_end=close_at_end)
    res["carried"] = []
    return {"frame": w, "res": res}


def _stats(x: np.ndarray, pnl: np.ndarray, capital: np.ndarray, tr: pd.DataFrame, side: int) -> dict:
    eq = np.cumsum(pnl); dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    peak = float(capital.max()) if len(capital) else 0.0; avg = float(capital.mean()) if len(capital) else 0.0; total = float(pnl.sum())
    return {"trades": len(tr), "total_ret": total, "hit": float((tr["ret"] > 0).mean()) if len(tr) else np.nan,
            "avg_ret": float(tr["ret"].mean()) if len(tr) else np.nan, "avg_held": float(tr["held"].mean()) if len(tr) else np.nan,
            "sharpe": float(pnl.mean() / pnl.std() * ANN) if len(pnl) > 1 and pnl.std() > 0 else np.nan,
            "max_dd": dd, "max_units": int(round(peak)), "peak_capital": peak, "avg_capital": avg,
            "ret_peak": total / peak if peak > 0 else np.nan, "ret_daywt": total / avg if avg > 0 else np.nan,
            "time_in_market": float((capital > 0).mean()) if len(capital) else 0.0,
            "armed_exits": float(tr["armed_exit"].mean()) if len(tr) else np.nan,
            "bh_ret": float((x[-1] - x[0]) * side) if len(x) else 0.0, "days": len(x),
            "capitulations": int(tr["capitulated"].sum()) if len(tr) else 0}


def run_versioned(sf: pd.DataFrame, versions: list[tuple[str, StrategyParams]], start: str, end: str | None,
                  close_at_end: bool) -> dict:
    """Live ledger with forward-only strategy changes.

    versions: [(effective_date, params), ...] sorted; the first applies from `start`. Open units are carried
    across a change (entry price, size and armed flag kept); the new rule governs exits and new entries.
    """
    m = sf["date"].to_numpy() >= start
    if end:
        m &= sf["date"].to_numpy() <= end
    w = sf[m].reset_index(drop=True)
    if len(w) < 1:
        return {"frame": w, "res": None}
    dates = w["date"].to_numpy()
    # segment boundaries: version dates strictly inside the window
    bounds = [0]
    for dte, _ in versions[1:]:
        k = int(np.searchsorted(dates, dte))
        if 0 < k < len(w) and k != bounds[-1]:
            bounds.append(k)
    bounds.append(len(w))
    # params per segment: the version in force at the segment's first date
    def params_at(d):
        cur = versions[0][1]
        for dte, pp in versions:
            if dte <= d:
                cur = pp
        return cur
    pnl, cap, trades, carried_dates = [], [], [], []
    init: list[dict] = []
    offset = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = w.iloc[a:b].reset_index(drop=True)
        pp = params_at(dates[a])
        last = b == len(w)
        res = run(seg, pp, close_at_end=(close_at_end and last), init_open=init)
        pnl.append(res["pnl"]); cap.append(res["capital"])
        for tr in res["trades"].to_dict(orient="records"):
            tr["entry"] = tr["entry"] + offset if tr["entry"] >= 0 else tr["entry"]  # carried: negative stays, resolved below
            tr["exit"] += offset
            trades.append(tr)
        # carry open units into the next segment: express their entry as absolute window index
        init = []
        for q in res["open"]:
            abs_i = q["i"] + offset if q["i"] >= 0 else q["abs"]  # carried-in units keep their absolute index
            init.append({"i": abs_i - b, "x0": q["x0"], "size": q["size"], "armed": q.get("armed", False), "z0": q.get("z0"), "abs": abs_i})
        offset = b
    # resolve carried (negative) entries: they are absolute indices shifted by their segment start; recompute
    # from the running record: a carried position's absolute index is stored in "abs" on init dicts. Trades
    # closed from carried positions have entry < 0 == (abs - segment_start); we re-derive via the positive path:
    pnl = np.concatenate(pnl); cap = np.concatenate(cap)
    tr = pd.DataFrame(trades)
    for c in ("entry", "exit", "held", "ret", "size", "pnl", "armed_exit", "capitulated"):
        if c not in tr.columns:
            tr[c] = pd.Series(dtype=float if c in ("ret", "size", "pnl") else (bool if c in ("armed_exit", "capitulated") else int))
    if len(tr):
        # negative entries: entry_abs = exit - held (held is counted in bars, continuous across segments)
        neg = tr["entry"] < 0
        tr.loc[neg, "entry"] = tr.loc[neg, "exit"] - tr.loc[neg, "held"]
        tr["entry"] = tr["entry"].clip(lower=0).astype(int)
    x = w["x"].to_numpy() * versions[0][1].side
    opens = [{"i": (q["abs"] if "abs" in q else q["i"]), "x0": q["x0"], "size": q["size"], "armed": q.get("armed", False)} for q in init]
    for q in opens:
        q["i"] = int(max(q["i"], 0))
    res = {"trades": tr, "pnl": pnl, "equity": np.cumsum(pnl), "capital": cap, "open": opens,
           "stats": _stats(x, pnl, cap, tr, versions[0][1].side), "carried": []}
    return {"frame": w, "res": res}


def trades_table(w: pd.DataFrame, res: dict, sym: str, side: int, reason_end: str) -> list[dict]:
    out = []
    d = w["date"].to_numpy(); c = np.exp(w["x"].to_numpy() * 1.0)
    for t in res["trades"].itertuples():
        reason = getattr(t, "reason", None) if hasattr(t, "reason") and getattr(t, "reason") not in (None, "end") else None
        if reason is None:
            reason = "capitulate" if t.capitulated else ("armed" if t.armed_exit else ("timeout" if t.held >= 1 and not t.armed_exit and t.exit < len(w) - 1 else reason_end))
        out.append({"symbol": sym, "side": side, "entry": d[t.entry], "exit": d[t.exit], "entry_px": float(c[t.entry]), "exit_px": float(c[t.exit]),
                    "size": float(t.size), "held": int(t.held), "ret": float(t.ret), "pnl": float(t.pnl), "reason": reason})
    return out


def sharpe(p: np.ndarray) -> float | None:
    p = np.asarray(p, float)
    return float(p.mean() / p.std() * ANN) if len(p) > 1 and p.std() > 0 else None


def side_book(symbol_runs: dict[str, dict]) -> dict:
    """Aggregate per-symbol runs (aligned by date) into exposure/equity series."""
    frames = {s: pd.DataFrame({"cap": r["res"]["capital"], "pnl": r["res"]["pnl"]}, index=r["frame"]["date"]) for s, r in symbol_runs.items() if r["res"] is not None}
    if not frames:
        return {"dates": [], "capital": [], "pnl": [], "equity": []}
    cap = pd.concat({s: f["cap"] for s, f in frames.items()}, axis=1).fillna(0).sum(axis=1)
    pnl = pd.concat({s: f["pnl"] for s, f in frames.items()}, axis=1).fillna(0).sum(axis=1)
    return {"dates": list(cap.index), "capital": cap.round(4).tolist(), "pnl": pnl.round(5).tolist(), "equity": pnl.cumsum().round(5).tolist()}


def combine(long: dict, short: dict) -> dict:
    idx = sorted(set(long["dates"]) | set(short["dates"]))
    L = pd.Series(long["capital"], index=long["dates"]).reindex(idx).fillna(0); S = pd.Series(short["capital"], index=short["dates"]).reindex(idx).fillna(0)
    PL = pd.Series(long["pnl"], index=long["dates"]).reindex(idx).fillna(0); PS = pd.Series(short["pnl"], index=short["dates"]).reindex(idx).fillna(0)
    pnl = PL + PS; eq = pnl.cumsum(); dd = float((eq.cummax() - eq).max()) if len(eq) else 0.0
    gross = L + S
    return {"dates": idx, "long": L.round(4).tolist(), "short": S.round(4).tolist(), "net": (L - S).round(4).tolist(), "gross": gross.round(4).tolist(),
            "pnl": pnl.round(5).tolist(), "equity": eq.round(5).tolist(), "eq_long": PL.cumsum().round(5).tolist(), "eq_short": PS.cumsum().round(5).tolist(),
            "summary": {"total": float(pnl.sum()), "total_long": float(PL.sum()), "total_short": float(PS.sum()), "sharpe": sharpe(pnl), "sharpe_long": sharpe(PL), "sharpe_short": sharpe(PS),
                        "max_dd": dd, "peak_gross": float(gross.max()) if len(gross) else 0.0, "avg_gross": float(gross.mean()) if len(gross) else 0.0,
                        "avg_net": float((L - S).mean()) if len(L) else 0.0, "days": len(idx),
                        "ret_peak": float(pnl.sum() / gross.max()) if len(gross) and gross.max() > 0 else None,
                        "ret_daywt": float(pnl.sum() / gross.mean()) if len(gross) and gross.mean() > 0 else None}}


def symbol_stats(r: dict) -> dict:
    if r["res"] is None:
        return {}
    st = r["res"]["stats"]
    keep = ("trades", "hit", "avg_held", "total_ret", "peak_capital", "avg_capital", "ret_peak", "ret_daywt", "max_dd", "sharpe", "bh_ret", "capitulations")
    return {k: (None if isinstance(st.get(k), float) and not np.isfinite(st[k]) else st.get(k)) for k in keep}
