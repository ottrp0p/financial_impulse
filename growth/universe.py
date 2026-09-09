"""Universe list + daily bar loading (yfinance via the repo's cached source)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from . import ROOT  # noqa: F401  (sets sys.path)
from impulse.bars import Bars
from impulse.sources import get_source

# Liquid US large caps (one share class per company), liquid mid/large growth names, ETFs, leveraged ETFs.
LARGE = """NVDA MSFT AAPL AMZN META GOOGL AVGO TSLA BRK-B JPM LLY V XOM UNH MA COST NFLX WMT PG JNJ HD ABBV
BAC ORCL CRM KO CVX MRK PEP AMD ADBE TMO CSCO ACN MCD ABT LIN WFC DIS INTU TXN QCOM CAT IBM GE AMGN
VZ PM ISRG NOW GS DHR NEE SPGI CMCSA UBER AXP LOW RTX BKNG PFE HON UNP T ETN BLK TJX COP SYK LMT
SCHW BSX VRTX PGR ADP AMAT MU C PANW GILD MDT DE ADI BMY MMC LRCX PLD FI SBUX KLAC CB MO SO ANET
INTC BA NKE PYPL ARM CRWD PLTR MRVL SNPS CDNS REGN DUK ZTS SHW ICE TGT CME EQIX WM MCK CL ORLY
APH MSI ITW PH FDX EMR NOC CTAS ABNB MAR COIN HOOD APP DASH SHOP TSM NVO ASML SMCI CVNA RDDT""".split()
ETF = "SPY QQQ IWM DIA XLK XLF XLE XLV XLY XLP XLI XLU XLB XLRE XLC SMH TLT GLD HYG EEM EFA".split()
LEVERAGED = "TQQQ SQQQ SOXL SOXS UPRO SPXU TNA TZA".split()

UNIVERSE = LARGE + ETF + LEVERAGED
FIT_BARS = 120


def universe(kind: str = "all") -> list[str]:
    return {"all": UNIVERSE, "large": LARGE, "etf": ETF, "leveraged": LEVERAGED}[kind]


def split_names(syms: list[str]) -> tuple[list[str], list[str]]:
    """Development / held-out split: A-M vs N-Z (ETFs and leveraged follow the same rule)."""
    dev = [s for s in syms if s[0].upper() <= "M"]
    hold = [s for s in syms if s[0].upper() > "M"]
    return dev, hold


def load_bars(symbols: list[str], start: datetime, end: datetime, cache_dir: Path | str = ROOT / "data",
              verbose: bool = True) -> dict[str, Bars]:
    src = get_source("yfinance", cache_dir=str(cache_dir))
    out: dict[str, Bars] = {}
    for s in symbols:
        try:
            b = src.fetch(s, "1d", start, end)
        except Exception as e:  # noqa: BLE001 — a missing ticker must not kill the run
            if verbose:
                print(f"  skip {s}: {e}")
            continue
        if len(b) < FIT_BARS + 60:
            if verbose:
                print(f"  skip {s}: only {len(b)} bars")
            continue
        out[s] = b
    return out


def adv_dollars(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Trailing median dollar volume, shifted so the value at t uses sessions <= t-1."""
    return (df["close"] * df["volume"]).rolling(n, min_periods=max(5, n // 2)).median().shift(1)
