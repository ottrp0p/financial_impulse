"""Sweep the confirmation-and-acceleration short on the short candidates and the top-50 large caps."""
from __future__ import annotations
import json, pickle, sys
from itertools import product
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from impulse.strategy import StrategyParams, run, signal_frame
from impulse.strategy_confirm import ConfirmParams, run_confirm

SHORTS = ["NKE","CAVA","ABNB","DASH","SPOT","RGTI","SPCE","RKLB","CAR","SBUX","RDDT","ACHR","SHAK"]
SP50 = ["NVDA","MSFT","AAPL","AMZN","META","GOOGL","AVGO","TSLA","BRK-B","JPM","LLY","V","XOM","UNH","MA","COST","NFLX","WMT","PG","JNJ",
        "HD","ABBV","BAC","ORCL","CRM","CVX","KO","CSCO","WFC","MRK","AMD","PEP","ACN","LIN","TMO","MCD","ADBE","ABT","PM","GE",
        "IBM","ISRG","CAT","GS","TXN","QCOM","INTU","NOW","AMGN","DIS"]
ANN = np.sqrt(252); sharpe = lambda p: float(np.mean(p)/np.std(p)*ANN) if np.std(p) > 0 else np.nan

def load(syms, band="field"):
    out = {}
    for s in syms:
        rr, lx, dates = pickle.load(open(f"out/roll/{s}.pkl", "rb")); sf = signal_frame(rr, lx, 10, band, "price")
        bars = sf["bar"].to_numpy(); out[s] = (sf, [dates[b] for b in bars])
    return out

def pooled(frames, fn):
    port = None; T = P = A = 0.0; npos = 0; tr = 0; nc = 0
    for s, (sf, _) in frames.items():
        res = fn(sf); st = res["stats"]
        port = res["pnl"] if port is None else port + res["pnl"][:len(port)]
        T += st["total_ret"]; P += st["peak_capital"]; A += st["avg_capital"]; npos += st["total_ret"] > 0; tr += st["trades"]; nc += st["capitulations"]
    eq = np.cumsum(port); dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(total=T, ret_peak=T/P if P else np.nan, ret_daywt=T/A if A else np.nan, sharpe=sharpe(port), port_dd=dd, n_pos=npos, n=len(frames), trades=tr, capitulations=nc)

def main():
    F13, F50 = load(SHORTS), load(SP50)
    SPACE = {"z_break": [1.0, 1.5], "z_enter": [0.0, -0.25], "step": [0.5, 0.75], "z_exit": [0.25], "accel": [1/3, 0.5, 2/3], "rearm": [False, True], "lookback": [20, 40]}
    rows = []
    for combo in product(*SPACE.values()):
        kw = dict(zip(SPACE, combo)); p = ConfirmParams(start=0.25, max_adds=5, max_hold=60, **kw)
        r13 = pooled(F13, lambda sf: run_confirm(sf, p)); r50 = pooled(F50, lambda sf: run_confirm(sf, p))
        rows.append({**{k: (None if v == np.inf else v) for k, v in kw.items()}, **{"s13_" + k: v for k, v in r13.items()}, **{"s50_" + k: v for k, v in r50.items()}})
    g = pd.DataFrame(rows); g["min_sharpe"] = g[["s13_sharpe", "s50_sharpe"]].min(axis=1); g.to_csv("out/short_confirm_sweep.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["z_break","z_enter","step","accel","rearm","lookback","s13_sharpe","s13_ret_peak","s13_port_dd","s13_n_pos","s13_trades","s50_sharpe","s50_ret_peak","s50_port_dd","s50_n_pos","s50_trades"]
    print("== top by min(Sharpe) across both universes =="); print(g.sort_values("min_sharpe", ascending=False)[cols].head(10).round(3).to_string(index=False))
    print("\n== top by 13-name Sharpe =="); print(g.sort_values("s13_sharpe", ascending=False)[cols].head(5).round(3).to_string(index=False))
    for k in SPACE:
        print(f"\n-- marginal by {k} --"); print(g.groupby(g[k].fillna("none"))[["s13_sharpe","s13_ret_peak","s50_sharpe","s50_ret_peak","s13_trades"]].median().round(3))
    # old best short rule for reference
    old = StrategyParams("price","field",10,1.0,0.5,1.5,100000,1.0,"fixed",1.0,6,-1,float("inf"),2.5)
    print("\n== reference: previous best short (fixed 1.0/0.5/1.5, capitulate 2.5) ==")
    print("13:", {k: round(v,3) for k, v in pooled(F13, lambda sf: run(sf, old)).items()}); print("50:", {k: round(v,3) for k, v in pooled(F50, lambda sf: run(sf, old)).items()})
    best = g.sort_values("min_sharpe", ascending=False).iloc[0]
    kw = {k: (bool(best[k]) if k == "rearm" else (int(best[k]) if k == "lookback" else float(best[k]))) for k in SPACE}
    p = ConfirmParams(start=0.25, max_adds=5, max_hold=60, **kw)
    # detail JSON for the artifact: chosen config on both universes + brownian band twin + old reference
    out = {"params": {k: (None if v == np.inf else v) for k, v in p.__dict__.items()}, "sweep": json.loads(g.round(4).to_json(orient="records")), "configs": {}, "space": {k: [None if v == np.inf else v for v in vs] for k, vs in SPACE.items()}}
    F13b = load(SHORTS, "brownian")
    for name, frames, fn in (("confirm · field · 13 shorts", F13, lambda sf: run_confirm(sf, p)), ("confirm · brownian · 13 shorts", F13b, lambda sf: run_confirm(sf, p)),
                             ("confirm · field · S&P 50", F50, lambda sf: run_confirm(sf, p)), ("previous best short · 13", F13, lambda sf: run(sf, old)), ("previous best short · S&P 50", F50, lambda sf: run(sf, old))):
        d = {"symbols": {}, "pooled": pooled(frames, fn)}
        for s, (sf, dates) in frames.items():
            res = fn(sf); tr = res["trades"]
            d["symbols"][s] = {"dates": dates, "close": np.round(np.exp(sf["x"].to_numpy()),3).tolist(), "q10": np.round(np.exp(sf["q10"].to_numpy()),3).tolist(), "q90": np.round(np.exp(sf["q90"].to_numpy()),3).tolist(),
                               "z": np.round(sf["z"].to_numpy(),3).tolist(), "equity": np.round(res["equity"],5).tolist(), "capital": np.round(res["capital"],4).tolist(),
                               "trades": tr.round(5).to_dict(orient="records"), "stats": {k: (None if isinstance(v,float) and not np.isfinite(v) else v) for k, v in res["stats"].items()}}
        d["pooled"] = {k: (None if isinstance(v,float) and not np.isfinite(v) else v) for k, v in d["pooled"].items()}
        out["configs"][name] = d
        print(f"\n{name}: {d['pooled']}")
    json.dump(out, open("out/short_confirm.json", "w"), allow_nan=False)

if __name__ == "__main__":
    main()
