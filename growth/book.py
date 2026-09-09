"""Daily open->close book: screen by liquidity at current equity, select by |mu/sigma| >= tau,
size with fractional Kelly on a Ledoit-Wolf covariance, fill open+slip / close-slip, compound."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .liquidity import eligible, position_caps, slippage
from .sizing import kelly_weights, ledoit_wolf, n_eff


@dataclass
class BookParams:
    tau: float = 0.10          # entry threshold on |mu_hat / sigma_hat|
    k: float = 0.5             # fractional Kelly
    n_max: int = 10            # max names per day
    L_max: float = 5.0         # gross leverage cap
    floor: float = 250e3
    E0: float = 500e3
    participation: float = 0.01
    min_adv: float = 5e7
    half_spread_bp: float = 1.0
    impact_coef: float = 0.1
    cov_window: int = 120
    margin_call: float = -0.25  # intraday marked loss (fraction of equity) that we flag
    fixed_equity: bool = False  # if True, caps/slippage use E0 every day (removes path dependence)


def run_book(pred: pd.DataFrame, p: BookParams) -> dict:
    """pred: rows (ts, symbol) with mu_hat, sigma_hat, oc, lo_oc, hi_oc, adv, sigma_bar."""
    pred = pred.sort_values(["ts", "symbol"])
    oc_panel = pred.pivot(index="ts", columns="symbol", values="oc")
    dates = oc_panel.index
    E = p.E0
    rows, trades = [], []
    by_day = {d: g for d, g in pred.groupby("ts")}
    for i, d in enumerate(dates):
        g = by_day[d]
        Eq = p.E0 if p.fixed_equity else E
        adv = g["adv"].to_numpy()
        elig = eligible(adv, Eq, p.participation, p.min_adv)
        score = (g["mu_hat"] / g["sigma_hat"]).to_numpy()
        pick = elig & np.isfinite(score) & (np.abs(score) >= p.tau)
        idx = np.flatnonzero(pick)
        idx = idx[np.argsort(-np.abs(score[idx]))][: p.n_max]
        day = {"ts": d, "equity": E, "ret": 0.0, "n": 0, "gross": 0.0, "n_eff": 0.0, "slip_bp": 0.0,
               "universe": int(elig.sum()), "worst_intraday": 0.0, "margin_flag": False}
        if len(idx) and i >= 20:
            sub = g.iloc[idx]
            syms = sub["symbol"].tolist()
            hist = oc_panel.iloc[max(0, i - p.cov_window):i][syms]
            hist = hist.dropna(how="all").fillna(0.0)
            if len(hist) < 20:
                Sigma = np.diag(sub["sigma_hat"].to_numpy() ** 2)
            else:
                Sigma = ledoit_wolf(hist.to_numpy())
            caps = position_caps(sub["adv"].to_numpy(), Eq, p.participation)
            f = kelly_weights(sub["mu_hat"].to_numpy(), Sigma, p.k, p.L_max, caps)
            keep = np.abs(f) > 1e-4
            if keep.any():
                f, sub, Sigma = f[keep], sub[keep], Sigma[np.ix_(keep, keep)]
                order = np.abs(f) * Eq
                slip = slippage(order, sub["adv"].to_numpy(), sub["sigma_bar"].to_numpy(), p.half_spread_bp, p.impact_coef)
                oc_s = np.expm1(sub["oc"].to_numpy())
                pnl = f * oc_s - np.abs(f) * 2 * slip
                ret = float(pnl.sum())
                worst = np.where(f > 0, np.expm1(sub["lo_oc"].to_numpy()), np.expm1(sub["hi_oc"].to_numpy()))
                worst_ret = float((f * worst).sum())
                E = E * (1 + ret)
                day.update({"ret": ret, "n": int(keep.sum()), "gross": float(np.abs(f).sum()), "n_eff": n_eff(f, Sigma),
                            "slip_bp": float((np.abs(f) * 2 * slip).sum() / max(np.abs(f).sum(), 1e-9) * 1e4),
                            "worst_intraday": worst_ret, "margin_flag": worst_ret <= p.margin_call})
                for s, fi, r_i, sl in zip(sub["symbol"], f, pnl, slip):
                    trades.append({"ts": d, "symbol": s, "f": fi, "pnl": r_i, "slip": sl})
        day["equity"] = E
        rows.append(day)
        if E <= p.floor and not p.fixed_equity:
            break
    daily = pd.DataFrame(rows)
    r = daily["ret"].to_numpy()
    ann = np.sqrt(252)
    stats = {
        "days": len(daily), "final_equity": E, "ruined": bool(E <= p.floor),
        "G": float(np.log1p(r).mean()), "V": float(np.log1p(r).var()),
        "sharpe": float(r.mean() / r.std() * ann) if r.std() > 0 else np.nan,
        "ann_ret": float(np.expm1(np.log1p(r).mean() * 252)),
        "max_dd": float((daily["equity"] / daily["equity"].cummax() - 1).min()),
        "avg_n": float(daily["n"].mean()), "avg_gross": float(daily["gross"].mean()),
        "avg_n_eff": float(daily.loc[daily["n"] > 0, "n_eff"].mean()) if (daily["n"] > 0).any() else 0.0,
        "avg_slip_bp": float(daily.loc[daily["n"] > 0, "slip_bp"].mean()) if (daily["n"] > 0).any() else 0.0,
        "margin_flags": int(daily["margin_flag"].sum()),
        "universe_start": int(daily["universe"].iloc[0]), "universe_end": int(daily["universe"].iloc[-1]),
        "days_traded_frac": float((daily["n"] > 0).mean()),
    }
    return {"daily": daily, "trades": pd.DataFrame(trades), "stats": stats, "params": asdict(p)}
