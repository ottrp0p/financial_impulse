"""Short-side parameter sweep over the short universe. Uses/extends out/roll/*.pkl caches."""
from __future__ import annotations
import pickle, sys
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from impulse.forecast import rolling_forecast
from impulse.sources import get_source
from impulse.strategy import StrategyParams, run, signal_frame

SYMS = ["NKE","CAVA","ABNB","DASH","SPOT","RGTI","SPCE","RKLB","CAR","SBUX","RDDT","ACHR","SHAK","CBRS"]
VIEW, FIT = 252, 120
end = datetime(2026, 8, 26); src = get_source("yfinance")
rrs, closes = {}, {}
for sym in SYMS:
    f = Path(f"out/roll/{sym}.pkl")
    if f.exists():
        rr, lx, _ = pickle.load(open(f, "rb"))
    else:
        try:
            b = src.fetch(sym, "1d", end - timedelta(days=565), end)
        except Exception as e:
            print("SKIP", sym, e, flush=True); continue
        if len(b) < FIT + 60:
            print("SKIP", sym, "only", len(b), "bars", flush=True); continue
        b = b.slice(max(0, len(b) - (FIT + VIEW + 10)), len(b)); lx = b.log_close
        rr = rolling_forecast(b, fit_bars=FIT, horizons=(5, 10), n_paths=2000, density_h=5,
                              heat_levels=np.linspace(lx.min()-0.03, lx.max()+0.03, 160), shrink_n0=50.0, sigma_mode="bar")
        pickle.dump((rr, lx, [d.strftime("%Y-%m-%d") for d in b.ts]), open(f, "wb")); print("rolled", sym, len(b), flush=True)
    rrs[sym], closes[sym] = rr, lx
print("universe:", list(rrs), flush=True)

SPACE = {"z_in": [1.0, 1.5, 2.0, 2.5], "z_out": [0.0, 0.5, 1.0, 1.5], "z_norm": [1.0, 1.5, None], "max_hold": [20, 40, 100000], "ratio": [1.5, 2.0, 3.0]}
sf_cache = {s: signal_frame(rrs[s], closes[s], 10, "field", "price") for s in rrs}
ANN = np.sqrt(252); sharpe = lambda p: float(p.mean()/p.std()*ANN) if p.std() > 0 else np.nan
rows = []
for zi, zo, zn, mh, r in product(*SPACE.values()):
    p = StrategyParams("price", "field", 10, zi, zo, zn, mh, 1.0, "decel", r, 4, -1)
    port = None; T = P = A = DD = 0.0; npos = 0; tr = 0; hits = []
    for s, sf in sf_cache.items():
        res = run(sf, p); st = res["stats"]
        port = res["pnl"] if port is None else port + res["pnl"][:len(port)]
        T += st["total_ret"]; P += st["peak_capital"]; A += st["avg_capital"]; DD += st["max_dd"]; npos += st["total_ret"] > 0; tr += st["trades"]; hits.append(st["hit"])
    eq = np.cumsum(port); pdd = float((np.maximum.accumulate(eq) - eq).max())
    rows.append(dict(z_in=zi, z_out=zo, z_norm=-1 if zn is None else zn, max_hold=mh, ratio=r, total=T, ret_peak=T/P if P else np.nan, ret_daywt=T/A if A else np.nan,
                     sharpe=sharpe(port), port_dd=pdd, dd_sum=DD, n_pos=npos, trades=tr, hit=np.nanmean(hits), calmar=T/pdd if pdd > 0 else np.nan))
g = pd.DataFrame(rows); g.to_csv("out/short_sweep.csv", index=False)
pd.set_option("display.width", 250)
print("\n== top by portfolio Sharpe =="); print(g.sort_values("sharpe", ascending=False).head(12).round(3).to_string(index=False))
print("\n== top by return on peak capital =="); print(g.sort_values("ret_peak", ascending=False).head(8).round(3).to_string(index=False))
print("\n== top by total/portfolio-DD (calmar-ish) =="); print(g.sort_values("calmar", ascending=False).head(8).round(3).to_string(index=False))
for k in ["z_in", "z_out", "z_norm", "max_hold", "ratio"]:
    print(f"\n-- marginal by {k}: median over other params --"); print(g.groupby(k)[["sharpe","ret_peak","port_dd","n_pos","trades"]].median().round(3))
