"""Layer-2 grid: symbols x intervals -> out/summary.md with go/no-go."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from impulse.evaluate import WalkForwardConfig, format_summary, verdict, walk_forward
from impulse.sources import get_source

WINDOWS = {"1h": (500, 100), "1d": (250, 60)}
LOOKBACK_DAYS = {"1h": 728, "1d": 3650}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,BTC-USD")
    ap.add_argument("--intervals", default="1h,1d")
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--out", default="out")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    end = datetime.strptime(a.end, "%Y-%m-%d") if a.end else datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    src = get_source(a.source)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# Layer-2 walk-forward summary", "",
             "Fixed settings: bandwidth = Silverman rule on fit-window log close; tau = fit/2; passage band = max(bandwidth, 3 sigma_bar);",
             f"windows (fit, test) = {WINDOWS}. Scores: gravity g(x), volume-only baseline, shuffled-g null.", ""]
    grid_rows = []
    for sym in a.symbols.split(","):
        for iv in a.intervals.split(","):
            fit_n, test_n = WINDOWS[iv]
            bars = src.fetch(sym, iv, end - timedelta(days=LOOKBACK_DAYS[iv]), end)
            pw, pooled, summ = walk_forward(bars, WalkForwardConfig(fit_bars=fit_n, test_bars=test_n))
            v = verdict(summ)
            pw.to_csv(out / f"{sym}_{iv}_eval.csv", index=False)
            grid_rows.append({"symbol": sym, "interval": iv, "windows": summ.attrs["n_windows"], **v})
            lines += [f"## {sym} {iv} ({len(bars)} bars)", "", "```", format_summary(summ), "```", f"verdict: {v}", ""]
            print(f"{sym} {iv}: {v}")
    grid = pd.DataFrame(grid_rows)
    n_pass = int(grid["pass"].sum())
    go = n_pass >= 4  # >= 2 of 3 symbols on both intervals
    lines.insert(4, f"**GO/NO-GO: {'GO' if go else 'NO-GO'}** — {n_pass}/{len(grid)} series pass (H1 & H4 & H5 & beats both nulls).\n\n" + "```\n" + grid.to_string(index=False) + "\n```\n")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n" + grid.to_string(index=False))
    print(f"\nGO/NO-GO: {'GO' if go else 'NO-GO'} ({n_pass}/{len(grid)})")


if __name__ == "__main__":
    main()
