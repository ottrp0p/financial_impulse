"""Long/short book on the top-50 S&P names: long rule + short rule on every name, daily exposures and P&L."""
from __future__ import annotations
import json, pickle, sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from impulse.forecast import rolling_forecast
from impulse.sources import get_source
from impulse.strategy import StrategyParams, run, signal_frame

SP50 = ["NVDA","MSFT","AAPL","AMZN","META","GOOGL","AVGO","TSLA","BRK-B","JPM","LLY","V","XOM","UNH","MA","COST","NFLX","WMT","PG","JNJ",
        "HD","ABBV","BAC","ORCL","CRM","CVX","KO","CSCO","WFC","MRK","AMD","PEP","ACN","LIN","TMO","MCD","ADBE","ABT","PM","GE",
        "IBM","ISRG","CAT","GS","TXN","QCOM","INTU","NOW","AMGN","DIS"]
VIEW, FIT = 252, 120
LONG = StrategyParams("price","field",10,1.0,1.5,1.0,100000,1.0,"decel",1.5,4, 1)
SHORT = StrategyParams("price","field",10,1.0,0.5,1.5,100000,1.0,"fixed",1.0,6,-1, float("inf"), 2.5)
ANN = np.sqrt(252); sharpe = lambda p: float(np.mean(p)/np.std(p)*ANN) if np.std(p) > 0 else float("nan")

def main():
    end = datetime(2026, 8, 26); src = get_source("yfinance"); Path("out/roll").mkdir(exist_ok=True, parents=True)
    per = {}; frames = {}
    for sym in SP50:
        f = Path(f"out/roll/{sym}.pkl")
        if f.exists():
            rr, lx, dates = pickle.load(open(f, "rb"))
        else:
            b = src.fetch(sym, "1d", end - timedelta(days=565), end); b = b.slice(max(0, len(b) - (FIT + VIEW + 10)), len(b)); lx = b.log_close
            rr = rolling_forecast(b, fit_bars=FIT, horizons=(5,10), n_paths=2000, density_h=5, heat_levels=np.linspace(lx.min()-0.03, lx.max()+0.03, 160), shrink_n0=50.0, sigma_mode="bar")
            dates = [d.strftime("%Y-%m-%d") for d in b.ts]; pickle.dump((rr, lx, dates), open(f, "wb")); print("rolled", sym, flush=True)
        sf = signal_frame(rr, lx, 10, "field", "price"); bars = sf["bar"].to_numpy(); ds = [dates[b] for b in bars]
        L, S = run(sf, LONG), run(sf, SHORT)
        df = pd.DataFrame({"long_cap": L["capital"], "short_cap": S["capital"], "long_pnl": L["pnl"], "short_pnl": S["pnl"],
                           "close": np.exp(sf["x"].to_numpy())}, index=pd.Index(ds, name="date"))
        frames[sym] = df
        per[sym] = {"long": {k: L["stats"][k] for k in ("trades","hit","total_ret","peak_capital","avg_capital","ret_peak","ret_daywt","max_dd","sharpe","bh_ret")},
                    "short": {k: S["stats"][k] for k in ("trades","hit","total_ret","peak_capital","avg_capital","ret_peak","ret_daywt","max_dd","sharpe","capitulations")},
                    "trades_long": L["trades"].round(5).to_dict(orient="records"), "trades_short": S["trades"].round(5).to_dict(orient="records"),
                    "dates": ds, "close": np.round(np.exp(sf["x"].to_numpy()),3).tolist(), "q10": np.round(np.exp(sf["q10"].to_numpy()),3).tolist(), "q90": np.round(np.exp(sf["q90"].to_numpy()),3).tolist()}
        for k in ("long","short"):
            per[sym][k] = {a: (None if isinstance(v,float) and not np.isfinite(v) else v) for a, v in per[sym][k].items()}
    book = pd.concat(frames, axis=1).T.groupby(level=1).sum().T   # sum across symbols by column
    book["net"] = book["long_cap"] - book["short_cap"]; book["gross"] = book["long_cap"] + book["short_cap"]
    book["pnl"] = book["long_pnl"] + book["short_pnl"]
    for c in ("long_pnl","short_pnl","pnl"): book["eq_" + c] = book[c].cumsum()
    eq = book["eq_pnl"].to_numpy(); dd = float((np.maximum.accumulate(eq) - eq).max())
    T = float(book["pnl"].sum()); TL = float(book["long_pnl"].sum()); TS = float(book["short_pnl"].sum())
    summary = {"n_names": len(frames), "days": len(book),
               "total": T, "total_long": TL, "total_short": TS,
               "sharpe": sharpe(book["pnl"]), "sharpe_long": sharpe(book["long_pnl"]), "sharpe_short": sharpe(book["short_pnl"]),
               "corr_long_short": float(np.corrcoef(book["long_pnl"], book["short_pnl"])[0,1]),
               "avg_long": float(book["long_cap"].mean()), "avg_short": float(book["short_cap"].mean()), "avg_net": float(book["net"].mean()), "avg_gross": float(book["gross"].mean()),
               "peak_long": float(book["long_cap"].max()), "peak_short": float(book["short_cap"].max()), "peak_gross": float(book["gross"].max()),
               "net_min": float(book["net"].min()), "net_max": float(book["net"].max()),
               "ret_peak_gross": T / float(book["gross"].max()), "ret_daywt_gross": T / float(book["gross"].mean()), "max_dd": dd,
               "frac_days_net_short": float((book["net"] < 0).mean()),
               "n_long_pos": int(sum(v["long"]["total_ret"] > 0 for v in per.values())), "n_short_pos": int(sum(v["short"]["total_ret"] > 0 for v in per.values()))}
    bh = pd.concat({s: np.log(f["close"]).diff().fillna(0) for s, f in frames.items()}, axis=1).mean(axis=1)
    summary["sharpe_bh_eqwt"] = sharpe(bh); summary["bh_eqwt_ret"] = float(bh.sum())
    out = {"symbols": list(frames), "long_params": {k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in LONG.__dict__.items()}, "short_params": {k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in SHORT.__dict__.items()},
           "summary": summary, "book": {"dates": list(book.index), **{c: np.round(book[c].to_numpy(), 4).tolist() for c in book.columns}}, "per": per}
    json.dump(out, open("out/ls_book.json", "w"), allow_nan=False)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}, indent=1))
    rows = [{"sym": s, **{"L_"+k: v["long"][k] for k in ("trades","total_ret","ret_peak","sharpe","bh_ret")}, **{"S_"+k: v["short"][k] for k in ("trades","total_ret","ret_peak","sharpe","capitulations")}} for s, v in per.items()]
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(3).to_string(index=False))

if __name__ == "__main__":
    main()
