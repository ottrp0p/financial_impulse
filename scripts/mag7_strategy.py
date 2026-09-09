"""Band-reversion strategy grid on the Mag 7 rolling forecasts. Writes out/strategy_grid.csv and out/strategy.json."""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from impulse.forecast import rolling_forecast
from impulse.sources import get_source
from impulse.strategy import StrategyParams, grid, run, signal_frame

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
VIEW, FIT = 252, 120
SPACE = {"signal": ["price", "gcenter"], "band": ["field", "brownian"], "h": [5, 10],
         "z_in": [1.0, 1.5, 2.0], "z_out": [0.5, 1.0, 1.5], "z_norm": [0.5, 1.0], "max_hold": [60]}


def main():
    end = datetime(2026, 8, 26)
    src = get_source("yfinance")
    cache = Path("out/rolling_cache.pkl")
    if cache.exists():
        rrs, closes, dates = pickle.load(open(cache, "rb"))
    else:
        rrs, closes, dates = {}, {}, {}
        for sym in MAG7:
            b = src.fetch(sym, "1d", end - timedelta(days=565), end)
            b = b.slice(max(0, len(b) - (FIT + VIEW + 10)), len(b))
            lx = b.log_close
            rrs[sym] = rolling_forecast(b, fit_bars=FIT, horizons=(5, 10), n_paths=2000, density_h=5,
                                        heat_levels=np.linspace(lx.min() - 0.03, lx.max() + 0.03, 160),
                                        shrink_n0=50.0, sigma_mode="bar")
            closes[sym] = lx
            dates[sym] = [d.strftime("%Y-%m-%d") for d in b.ts]
            print("rolled", sym)
        pickle.dump((rrs, closes, dates), open(cache, "wb"))

    g = grid(rrs, closes, SPACE)
    g.to_csv("out/strategy_grid.csv", index=False)
    keys = list(SPACE)
    agg = g.groupby(keys).agg(total_ret=("total_ret", "sum"), trades=("trades", "sum"), hit=("hit", "mean"),
                              sharpe=("sharpe", "median"), n_pos=("total_ret", lambda s: (s > 0).sum()),
                              avg_held=("avg_held", "mean"), max_dd=("max_dd", "max")).reset_index()
    agg["sharpe"] = agg["sharpe"].round(2)
    print("\n== configs ranked by pooled total return per unit (sum over 7 names) ==")
    print(agg.sort_values("total_ret", ascending=False).head(12).to_string(index=False))
    print("\n== worst ==")
    print(agg.sort_values("total_ret").head(5).to_string(index=False))
    # field vs brownian band, paired on all other params
    piv = agg.pivot_table(index=[k for k in keys if k != "band"], columns="band", values="total_ret").reset_index()
    piv["field_minus_bw"] = piv["field"] - piv["brownian"]
    print(f"\nfield band beats brownian band in {(piv['field_minus_bw'] > 0).mean():.0%} of paired configs; "
          f"mean diff {piv['field_minus_bw'].mean():+.3f} per unit (sum over names)")
    print("by signal:"); print(piv.groupby("signal")["field_minus_bw"].agg(["mean", lambda s: (s > 0).mean()]).round(3))
    print("\nby signal x band (median over configs of pooled total_ret):")
    print(agg.groupby(["signal", "band"])["total_ret"].agg(["median", "mean", "max"]).round(3))

    # detail for the best config and its brownian twin + a fixed reference config
    best = agg.sort_values("total_ret", ascending=False).iloc[0]
    ref = StrategyParams(signal="price", band="field", h=5, z_in=1.5, z_out=1.0, z_norm=0.5)
    out = {"space": SPACE, "agg": agg.round(4).to_dict(orient="records"), "detail": {}}
    for name, p in (("best", StrategyParams(**{k: best[k] for k in keys})),
                    ("best_brownian", StrategyParams(**{**{k: best[k] for k in keys}, "band": "brownian"})),
                    ("reference", ref), ("reference_brownian", StrategyParams(**{**ref.__dict__, "band": "brownian"}))):
        p.h = int(p.h); p.max_hold = int(p.max_hold)
        d = {"params": p.__dict__, "symbols": {}}
        for sym, rr in rrs.items():
            sf = signal_frame(rr, closes[sym], p.h, p.band, p.signal)
            res = run(sf, p)
            bars = sf["bar"].to_numpy()
            d["symbols"][sym] = {"dates": [dates[sym][b] for b in bars], "close": np.round(np.exp(closes[sym][bars]), 3).tolist(),
                                 "z": np.round(sf["z"].to_numpy(), 3).tolist(), "equity": np.round(res["equity"], 5).tolist(),
                                 "q10": np.round(np.exp(sf["q10"].to_numpy()), 3).tolist(), "q90": np.round(np.exp(sf["q90"].to_numpy()), 3).tolist(),
                                 "trades": res["trades"].round(5).to_dict(orient="records"), "stats": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in res["stats"].items()}}
        out["detail"][name] = d
        tot = sum(v["stats"]["total_ret"] for v in d["symbols"].values()); ntr = sum(v["stats"]["trades"] for v in d["symbols"].values())
        print(f"\n{name}: {p.__dict__} -> pooled ret {tot:+.3f} over {ntr} trades")
    Path("out/strategy.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
