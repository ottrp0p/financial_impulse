"""Mag 7, daily, last 120 bars: full-window field + 80/40 in/out split. Writes out/mag7.json."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from impulse.evaluate import split_test
from impulse.potential import fit_field
from impulse.sources import get_source

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
N, FIT, TEST = 120, 80, 40


def clean(a):
    return [None if (x is None or (isinstance(x, float) and not np.isfinite(x))) else x for x in a]


def main():
    end = datetime(2026, 8, 26)
    src = get_source("yfinance")
    out = {}
    for sym in MAG7:
        b = src.fetch(sym, "1d", end - timedelta(days=260), end)
        b = b.slice(len(b) - N, len(b))
        full = fit_field(b, tau=N / 2, min_n_eff=10)
        sp = split_test(b, FIT, TEST, min_levels=5, min_n_eff=10)
        t = full.table
        st = sp["field"].table
        out[sym] = {
            "dates": [d.strftime("%Y-%m-%d") for d in b.ts],
            "ohlc": {k: clean(getattr(b, k).round(3).tolist()) for k in ("open", "high", "low", "close")},
            "volume": clean(b.volume.tolist()),
            "full": {"levels": clean(np.exp(full.grid.levels).round(3).tolist()),
                     "band": float(np.exp(full.band_half_width) - 1),
                     **{c: clean(t[c].astype(float).where(t["valid"]).tolist()) for c in ("mass", "sigma", "mu", "U", "g", "residence_median")}},
            "split": {"levels": clean(np.exp(sp["grid"].levels).round(3).tolist()),
                      "g": clean(st["g"].astype(float).where(st["valid"]).tolist()),
                      "test_residence": clean(sp["level_stats"]["residence_median"].tolist()),
                      "test_n": clean(sp["level_stats"]["n_episodes"].astype(float).tolist()),
                      "fit_range": [float(b.close[:FIT].min()), float(b.close[:FIT].max())],
                      "frac_inside": float(np.mean((b.close[FIT:] >= b.close[:FIT].min()) & (b.close[FIT:] <= b.close[:FIT].max()))),
                      "metrics": sp["metrics"].replace({np.nan: None}).to_dict(orient="records")},
        }
        m = sp["metrics"].set_index("score")
        print(f"{sym}: {len(b)} bars {out[sym]['dates'][0]}..{out[sym]['dates'][-1]} | episodes={m.n_episodes.iloc[0]} | "
              f"gravity H1={m.h1_ratio['gravity']:.2f} p={m.h1_p['gravity']:.2g} H4={m.h4_rho['gravity']:.2f} | "
              f"volume H1={m.h1_ratio['volume']:.2f} H4={m.h4_rho['volume']:.2f} | shuffled H4={m.h4_rho['shuffled']:.2f}")
    Path("out").mkdir(exist_ok=True)
    Path("out/mag7.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
