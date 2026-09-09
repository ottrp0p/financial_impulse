"""CSV connector: one file per (symbol, interval) or an explicit path."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..bars import Bars

COLUMN_ALIASES = {
    "date": "ts", "datetime": "ts", "timestamp": "ts", "time": "ts",
    "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vol": "volume",
}


def read_bars_csv(path: str | Path, symbol: str = "?", interval: str = "?") -> Bars:
    df = pd.read_csv(path)
    df.columns = [COLUMN_ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
    if "ts" not in df.columns:
        raise ValueError(f"{path}: no timestamp column (date/datetime/timestamp/ts)")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return Bars(df, symbol=symbol, interval=interval)


class CSVSource:
    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)

    def fetch(self, symbol: str, interval: str, start: datetime, end: datetime) -> Bars:
        path = self.root / f"{symbol}_{interval}.csv"
        bars = read_bars_csv(path, symbol, interval)
        df = bars.df.loc[pd.Timestamp(start, tz="UTC"): pd.Timestamp(end, tz="UTC")]
        return Bars(df.copy(), symbol, interval)
