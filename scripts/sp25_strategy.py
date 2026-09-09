"""Universe runner: rolling forecasts (cached per symbol) + decel / fixed / brownian configs with Sharpe.
Usage: sp25_strategy.py [--symbols A,B,C] [--side 1|-1] [--out out/x.json]"""
from __future__ import annotations
import json, pickle, sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from impulse.forecast import rolling_forecast
from impulse.sources import get_source
from impulse.strategy import StrategyParams, run, signal_frame

# approximate top-25 S&P 500 by index weight (one share class per company)
DEFAULT = ["NVDA","MSFT","AAPL","AMZN","META","GOOGL","AVGO","TSLA","BRK-B","JPM","LLY","V","XOM","UNH","MA",
        "COST","NFLX","WMT","PG","JNJ","HD","ABBV","BAC","ORCL","CRM","SPY"]
VIEW, FIT = 252, 120
CFGS = {"decel · field": StrategyParams("price","field",10,1.0,1.5,1.0,100000,1.0,"decel",1.5,4),
        "decel · brownian": StrategyParams("price","brownian",10,1.0,1.5,1.0,100000,1.0,"decel",1.5,4),
        "fixed · field": StrategyParams("price","field",10,1.0,1.5,1.0,100000,1.0,"fixed",1.0,4)}
ANN = np.sqrt(252)

def sharpe(p): p = np.asarray(p, float); return float(p.mean()/p.std()*ANN) if p.std() > 0 else float("nan")

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", default=",".join(DEFAULT)); ap.add_argument("--side", type=int, default=1); ap.add_argument("--out", default="out/sp25.json"); ap.add_argument("--params", default=None, help="z_in,z_out,z_norm|none,max_hold,ratio"); ap.add_argument("--cap", type=float, default=float("inf")); ap.add_argument("--capit", type=float, default=float("inf"))
    a = ap.parse_args(); SP25 = a.symbols.split(",")
    for p in CFGS.values(): p.side = a.side; p.max_units = a.cap; p.capitulate = a.capit; p.max_stack = 6
    if a.params:
        zi, zo, zn, mh, r = a.params.split(","); zn = None if zn == "none" else float(zn)
        for p in CFGS.values():
            p.z_in, p.z_out, p.z_norm, p.max_hold = float(zi), float(zo), zn, int(mh)
            if p.sizing == "decel": p.ratio = float(r)
    end = datetime(2026, 8, 26); src = get_source("yfinance")
    Path("out/roll").mkdir(parents=True, exist_ok=True)
    rrs, closes, dates = {}, {}, {}
    old = pickle.load(open("out/rolling_cache.pkl","rb")) if Path("out/rolling_cache.pkl").exists() else ({}, {}, {})
    for sym in SP25:
        f = Path(f"out/roll/{sym}.pkl")
        if sym in old[0]:
            rrs[sym], closes[sym], dates[sym] = old[0][sym], old[1][sym], old[2][sym]
        elif f.exists():
            rrs[sym], closes[sym], dates[sym] = pickle.load(open(f, "rb"))
        else:
            b = src.fetch(sym, "1d", end - timedelta(days=565), end)
            b = b.slice(max(0, len(b) - (FIT + VIEW + 10)), len(b)); lx = b.log_close
            rr = rolling_forecast(b, fit_bars=FIT, horizons=(5, 10), n_paths=2000, density_h=5,
                                  heat_levels=np.linspace(lx.min()-0.03, lx.max()+0.03, 160), shrink_n0=50.0, sigma_mode="bar")
            rrs[sym], closes[sym], dates[sym] = rr, lx, [d.strftime("%Y-%m-%d") for d in b.ts]
            pickle.dump((rr, lx, dates[sym]), open(f, "wb"))
            print("rolled", sym, flush=True)
    rng = np.random.default_rng(0)
    out = {"symbols": SP25, "side": a.side, "configs": {}}
    for name, p in CFGS.items():
        d = {"params": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in p.__dict__.items()}, "symbols": {}}; port = None; bh_port = None
        for sym in SP25:
            rr = rrs[sym]; sf = signal_frame(rr, closes[sym], p.h, p.band, p.signal); res = run(sf, p); st = res["stats"]; tr = res["trades"]
            x = sf["x"].to_numpy(); bars = sf["bar"].to_numpy()
            st["sharpe_bh"] = sharpe(np.diff(x, prepend=x[0]) * a.side)
            nulls = []
            for _ in range(1000):
                tot = 0.0
                for t in tr.itertuples():
                    e = rng.integers(0, len(x) - t.held - 1); tot += (x[e+t.held]-x[e]) * t.size * a.side
                nulls.append(tot)
            st["null_p"] = float(np.mean(np.array(nulls) >= st["total_ret"])); st["null_mean"] = float(np.mean(nulls))
            port = res["pnl"] if port is None else port + res["pnl"][:len(port)]
            bhr = np.diff(x, prepend=x[0]) * a.side; bh_port = bhr if bh_port is None else bh_port + bhr[:len(bh_port)]
            d["symbols"][sym] = {"dates":[dates[sym][b] for b in bars], "close":np.round(np.exp(x),3).tolist(),
                "q10":np.round(np.exp(sf["q10"].to_numpy()),3).tolist(), "q90":np.round(np.exp(sf["q90"].to_numpy()),3).tolist(),
                "z":np.round(sf["z"].to_numpy(),3).tolist(), "equity":np.round(res["equity"],5).tolist(), "capital":np.round(res["capital"],4).tolist(),
                "trades":tr.round(5).to_dict(orient="records"), "stats":{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in st.items()}}
        S = d["symbols"]; T = sum(v["stats"]["total_ret"] for v in S.values()); P = sum(v["stats"]["peak_capital"] for v in S.values()); A = sum(v["stats"]["avg_capital"] for v in S.values())
        d["portfolio"] = {"sharpe": sharpe(port), "sharpe_bh": sharpe(bh_port / len(SP25)), "total": T, "peak": P, "avg": A,
                          "equity": np.round(np.cumsum(port), 4).tolist(), "n_pos": int(sum(v["stats"]["total_ret"] > 0 for v in S.values())),
                          "n_null": int(sum(v["stats"]["null_p"] < 0.05 for v in S.values()))}
        out["configs"][name] = d
        print(f"\n{name}: total {T:+.2f} | peak {P:.1f} -> {T/P:+.1%} | daywt {A:.1f} -> {T/A:+.1%} | portfolio Sharpe {d['portfolio']['sharpe']:.2f} (B&H eq-wt {d['portfolio']['sharpe_bh']:.2f}) | +ve {d['portfolio']['n_pos']}/25 | beat null {d['portfolio']['n_null']}/25")
        rows = [{"sym": s, **{k: v["stats"][k] for k in ("trades","hit","avg_held","peak_capital","avg_capital","total_ret","ret_peak","ret_daywt","max_dd","sharpe","sharpe_bh","bh_ret","null_p")}} for s, v in S.items()]
        pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(3).to_string(index=False))
    json.dump(out, open(a.out, "w"), allow_nan=False)

if __name__ == "__main__":
    main()
