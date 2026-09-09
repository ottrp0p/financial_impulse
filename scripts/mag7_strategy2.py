"""Looser exits + accelerating/decelerating sizing sweep. Uses out/rolling_cache.pkl. Writes out/strategy2_grid.csv."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from impulse.strategy import StrategyParams, run, signal_frame

rrs, closes, dates = pickle.load(open("out/rolling_cache.pkl", "rb"))
SPACE = {"band": ["field", "brownian"], "h": [5, 10], "z_in": [1.0, 1.5], "z_out": [0.0, 0.5, 1.0, 1.5],
         "z_norm": [0.5, 1.0, None], "max_hold": [60, 100000]}
SIZING = [("fixed", 1.0), ("accel", 1.5), ("accel", 2.0), ("decel", 1.5), ("decel", 2.0)]
from itertools import product
cache = {}
rows = []
for band, h, zi, zo, zn, mh in product(*SPACE.values()):
    for sizing, ratio in SIZING:
        p = StrategyParams("price", band, h, zi, zo, zn, mh, 1.0, sizing, ratio)
        for sym, rr in rrs.items():
            ck = (sym, h, band)
            if ck not in cache:
                cache[ck] = signal_frame(rr, closes[sym], h, band, "price")
            st = run(cache[ck], p)["stats"]
            rows.append({"band": band, "h": h, "z_in": zi, "z_out": zo, "z_norm": -1 if zn is None else zn, "max_hold": mh,
                         "sizing": f"{sizing}{'' if sizing=='fixed' else ratio}", "symbol": sym, **st})
g = pd.DataFrame(rows); g.to_csv("out/strategy2_grid.csv", index=False)
keys = ["band", "h", "z_in", "z_out", "z_norm", "max_hold", "sizing"]
agg = g.groupby(keys).agg(total=("total_ret", "sum"), peak_cap=("peak_capital", "sum"), avg_cap=("avg_capital", "sum"),
                          trades=("trades", "sum"), hit=("hit", "mean"), held=("avg_held", "mean"), n_pos=("total_ret", lambda s: (s > 0).sum()),
                          armed=("armed_exits", "mean"), tim=("time_in_market", "mean"), sharpe=("sharpe", "median")).reset_index()
agg["ret_peak"] = agg.total / agg.peak_cap; agg["ret_daywt"] = agg.total / agg.avg_cap
agg.to_csv("out/strategy2_agg.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
def show(d, title, n=10):
    print(f"\n== {title} ==")
    print(d.head(n).round(3).to_string(index=False))
fixed = agg[agg.sizing == "fixed"]
show(fixed.sort_values("total", ascending=False), "FIXED sizing: top by total P&L (units)")
show(fixed.sort_values("ret_peak", ascending=False), "FIXED sizing: top by return on PEAK capital")
show(fixed.sort_values("ret_daywt", ascending=False), "FIXED sizing: top by return on DAY-WEIGHTED capital")
print("\n== exit looseness, fixed sizing, field band, h=10, z_in=1.0, no timeout: by (z_out, z_norm) ==")
print(fixed[(fixed.band=="field")&(fixed.h==10)&(fixed.z_in==1.0)&(fixed.max_hold>60)].pivot_table(index="z_out", columns="z_norm", values=["total","ret_peak","ret_daywt","held"]).round(2))
print("\n== timeout effect (fixed, field, h=10, z_in=1): mean over exit configs ==")
print(fixed[(fixed.band=="field")&(fixed.h==10)&(fixed.z_in==1.0)].groupby("max_hold")[["total","ret_peak","ret_daywt","held","armed"]].mean().round(3))
print("\n== SIZING comparison: median over all exit configs (field band) ==")
print(agg[agg.band=="field"].groupby("sizing")[["total","peak_cap","avg_cap","ret_peak","ret_daywt","hit","n_pos"]].median().round(3))
print("\n== SIZING: best config per sizing scheme by return on day-weighted capital (field band) ==")
print(agg[agg.band=="field"].sort_values("ret_daywt", ascending=False).groupby("sizing").head(1).round(3).to_string(index=False))
print("\n== SIZING: best config per sizing scheme by return on PEAK capital (field band) ==")
print(agg[agg.band=="field"].sort_values("ret_peak", ascending=False).groupby("sizing").head(1).round(3).to_string(index=False))
piv = agg.pivot_table(index=[k for k in keys if k!="band"], columns="band", values="total").reset_index()
print(f"\nfield beats brownian in {((piv.field-piv.brownian)>0).mean():.0%} of paired configs, mean diff {(piv.field-piv.brownian).mean():+.2f}")
