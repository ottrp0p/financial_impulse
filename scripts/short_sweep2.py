"""Short sweep v3: loose entry, heavy decel, loose exit, 5-unit cap, 2.5-unit capitulation."""
from __future__ import annotations
import pickle, sys
from itertools import product
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from impulse.strategy import StrategyParams, run, signal_frame
SYMS = ["NKE","CAVA","ABNB","DASH","SPOT","RGTI","SPCE","RKLB","CAR","SBUX","RDDT","ACHR","SHAK"]
sf = {}
for s in SYMS:
    rr, lx, _ = pickle.load(open(f"out/roll/{s}.pkl", "rb")); sf[s] = signal_frame(rr, lx, 10, "field", "price")
ANN = np.sqrt(252); sharpe = lambda p: float(p.mean()/p.std()*ANN) if p.std() > 0 else np.nan
SPACE = {"z_in": [0.5, 1.0], "z_out": [0.0, 0.5, 1.0, 1.5], "z_norm": [1.0, 1.5, None], "max_hold": [40, 100000], "ratio": [2.0, 3.0, 4.0], "cap": [5.0, np.inf], "capit": [2.5, np.inf]}
rows = []
for zi, zo, zn, mh, r, cap, cp in product(*SPACE.values()):
    p = StrategyParams("price", "field", 10, zi, zo, zn, mh, 1.0, "decel", r, 6, -1, cap, cp)
    port = None; T = P = A = 0.0; npos = 0; tr = 0; nc = 0
    for s, f in sf.items():
        res = run(f, p); st = res["stats"]
        port = res["pnl"] if port is None else port + res["pnl"][:len(port)]
        T += st["total_ret"]; P += st["peak_capital"]; A += st["avg_capital"]; npos += st["total_ret"] > 0; tr += st["trades"]; nc += st["capitulations"]
    eq = np.cumsum(port); pdd = float((np.maximum.accumulate(eq) - eq).max())
    rows.append(dict(z_in=zi, z_out=zo, z_norm=-1 if zn is None else zn, max_hold=mh, ratio=r, cap=cap, capit=cp, total=T, ret_peak=T/P if P else np.nan,
                     ret_daywt=T/A if A else np.nan, sharpe=sharpe(port), port_dd=pdd, n_pos=npos, trades=tr, capitulations=nc, calmar=T/pdd if pdd>0 else np.nan))
g = pd.DataFrame(rows); g.to_csv("out/short_sweep2.csv", index=False)
pd.set_option("display.width", 260)
print("== top by Sharpe =="); print(g.sort_values("sharpe", ascending=False).head(10).round(3).to_string(index=False))
print("\n== top by Sharpe with cap=5 & capit=2.5 (the requested regime) =="); print(g[(g.cap==5)&(g.capit==2.5)].sort_values("sharpe", ascending=False).head(10).round(3).to_string(index=False))
print("\n== top by return on peak (cap=5 & capit=2.5) =="); print(g[(g.cap==5)&(g.capit==2.5)].sort_values("ret_peak", ascending=False).head(6).round(3).to_string(index=False))
for k in ["z_in","z_out","z_norm","max_hold","ratio","cap","capit"]:
    print(f"\n-- marginal by {k} --"); print(g.groupby(k)[["sharpe","ret_peak","port_dd","n_pos","trades","capitulations"]].median().round(3))
