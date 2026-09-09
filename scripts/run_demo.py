"""End-to-end on one symbol: fit the field on the most recent window, plot it, run walk-forward eval."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from impulse.evaluate import WalkForwardConfig, format_summary, verdict, walk_forward
from impulse.plot import plot_field
from impulse.potential import fit_field
from impulse.sources import get_source

# window sizes are fixed per interval (bars). Bandwidth = Silverman rule, tau = fit/2, band = 3 sigma.
WINDOWS = {"1h": (500, 100), "1d": (250, 60)}
LOOKBACK_DAYS = {"1h": 728, "1d": 3650}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--interval", default="1h", choices=list(WINDOWS))
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--out", default="out")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    a = ap.parse_args()

    end = datetime.strptime(a.end, "%Y-%m-%d") if a.end else datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    bars = get_source(a.source).fetch(a.symbol, a.interval, end - timedelta(days=LOOKBACK_DAYS[a.interval]), end)
    fit_n, test_n = WINDOWS[a.interval]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    recent = bars.slice(len(bars) - fit_n, len(bars))
    fld = fit_field(recent, tau=fit_n / 2)
    fig = plot_field(recent, fld)
    png = out / f"{a.symbol}_{a.interval}_field.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    fld.table.to_csv(out / f"{a.symbol}_{a.interval}_field.csv")

    pw, pooled, summ = walk_forward(bars, WalkForwardConfig(fit_bars=fit_n, test_bars=test_n))
    pw.to_csv(out / f"{a.symbol}_{a.interval}_eval.csv", index=False)
    print(f"{a.symbol} {a.interval}: {len(bars)} bars, fit={fit_n} test={test_n}")
    print(format_summary(summ))
    print("verdict:", verdict(summ))
    print("wrote", png)


if __name__ == "__main__":
    main()
