"""Mag 7, daily, 365-day view: rolling 120-bar field, h-step predictive densities vs Brownian. Writes out/mag7_365.json."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from impulse.forecast import rolling_forecast, skill_summary
from impulse.sources import get_source

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
VIEW, FIT, HORIZONS, DH = 252, 120, (1, 5, 10), 5
SHRINK, SIGMA = 50.0, "bar"


def u8(a, lo, hi):
    a = np.asarray(a, float)
    q = np.where(np.isfinite(a), 1 + np.clip((a - lo) / (hi - lo), 0, 1) * 254, 0)
    return q.astype(np.uint8).tolist()


def main():
    end = datetime(2026, 8, 26)
    src = get_source("yfinance")
    out = {"meta": {"view": VIEW, "fit": FIT, "horizons": HORIZONS, "density_h": DH, "shrink_n0": SHRINK, "sigma_mode": SIGMA}}
    pooled = []
    for sym in MAG7:
        b = src.fetch(sym, "1d", end - timedelta(days=565), end)
        need = FIT + VIEW + max(HORIZONS)
        b = b.slice(max(0, len(b) - need), len(b))
        lx = b.log_close
        levels = np.linspace(lx.min() - 0.03, lx.max() + 0.03, 160)
        r = rolling_forecast(b, fit_bars=FIT, horizons=HORIZONS, n_paths=2000, density_h=DH,
                             heat_levels=levels, shrink_n0=SHRINK, sigma_mode=SIGMA)
        fc = r.forecasts
        sk = skill_summary(fc)
        pooled.append(fc.assign(symbol=sym))
        # view = last VIEW bars; rolling rows are indexed by t (bar index where forecast is issued = t-1)
        v0 = len(b) - VIEW
        dates = [d.strftime("%Y-%m-%d") for d in b.ts[v0:]]
        idx_of_t = {t: i for i, t in enumerate(range(FIT, len(b) - max(HORIZONS) + 1))}
        # per view date: density column for the forecast that *targets* this date (issued DH days earlier)
        n_lv = len(levels)
        dens = np.zeros((VIEW, n_lv)); gmat = np.full((VIEW, n_lv), np.nan)
        qm = np.full((VIEW, 3), np.nan); qb = np.full((VIEW, 3), np.nan)
        f5 = fc[fc.h == DH].set_index("t")
        for j in range(VIEW):
            bar = v0 + j                      # absolute bar index of this view date
            t_issue = bar - DH + 1            # forecast issued with x0 = close at bar t_issue-1, targets bar t_issue-1+DH = bar
            if t_issue in idx_of_t:
                col = r.heat_density[idx_of_t[t_issue]]
                dens[j] = col / col.max() if col.max() > 0 else 0
                if t_issue in f5.index:
                    row = f5.loc[t_issue]
                    qm[j] = [row.m_q10, row.m_q50, row.m_q90]; qb[j] = [row.b_q10, row.b_q50, row.b_q90]
            t_now = bar + 1                   # field fitted on bars < t_now, i.e. including this date
            if t_now in idx_of_t:
                gmat[j] = r.heat_g[idx_of_t[t_now]]
        out[sym] = {
            "dates": dates,
            "close": np.round(b.close[v0:], 3).tolist(),
            "high": np.round(b.high[v0:], 3).tolist(), "low": np.round(b.low[v0:], 3).tolist(),
            "levels": np.round(np.exp(levels), 3).tolist(),
            "density": u8(dens, 0, 1), "gravity": u8(gmat, -1.5, 1.5),
            "qm": np.where(np.isfinite(qm), np.round(np.exp(qm), 3), None).tolist(),
            "qb": np.where(np.isfinite(qb), np.round(np.exp(qb), 3), None).tolist(),
            "skill": sk.round(4).to_dict(orient="records"),
        }
        print(sym, len(b), "bars;", " | ".join(f"h{int(x.h)} gain={x.loglik_gain:+.3f} [{x.ci_lo:+.3f},{x.ci_hi:+.3f}] crps={x.crps_skill:+.3f} cov={x.cov80_model:.2f}/{x.cov80_brownian:.2f} w={x.width_ratio:.2f}" for x in sk.itertuples()))
    allfc = pd.concat(pooled)
    out["meta"]["pooled"] = skill_summary(allfc).round(4).to_dict(orient="records")
    print("POOLED"); print(skill_summary(allfc).round(3).to_string(index=False))
    Path("out/mag7_365.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
