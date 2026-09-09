"""yfinance connector with a parquet cache. The only place yfinance is imported."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..bars import Bars


class YFinanceSource:
    def __init__(self, cache_dir: str | Path = "data", use_cache: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

    def _cache_path(self, symbol: str, interval: str, start: datetime, end: datetime) -> Path:
        tag = f"{symbol}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.parquet".replace("/", "-")
        return self.cache_dir / tag

    def fetch(self, symbol: str, interval: str, start: datetime, end: datetime) -> Bars:
        path = self._cache_path(symbol, interval, start, end)
        if self.use_cache and path.exists():
            return Bars(pd.read_parquet(path), symbol, interval)

        import yfinance as yf  # local import: keeps the estimator modules free of it

        raw = yf.download(symbol, start=start, end=end, interval=interval,
                          auto_adjust=True, progress=False, threads=False)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {symbol} {interval} {start}..{end}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
        df = df.dropna()
        df = df[df["volume"] > 0]  # yfinance emits zero-volume placeholder rows intraday
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "ts"
        df = df[~df.index.duplicated(keep="last")].sort_index()
        bars = Bars(df, symbol, interval)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            bars.df.to_parquet(path)
        return bars
