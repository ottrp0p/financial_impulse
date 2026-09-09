"""Canonical bar schema and the data-source contract.

Everything downstream (mass, dynamics, passage, evaluate) consumes `Bars` only.
Third-party data libraries are imported exclusively inside `impulse.sources`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class Bars:
    """OHLCV bars with a fixed schema.

    `df`: UTC DatetimeIndex named 'ts'; float64 columns open/high/low/close/volume.
    Optional extra columns (vwap, trades, ...) are allowed and ignored by estimators.
    """

    df: pd.DataFrame
    symbol: str = "?"
    interval: str = "?"
    _validated: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.validate()

    # ---- schema -----------------------------------------------------------
    def validate(self) -> "Bars":
        df = self.df
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Bars missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Bars index must be a DatetimeIndex")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "ts"
        if not df.index.is_monotonic_increasing:
            raise ValueError("Bars index must be monotonic increasing")
        if df.index.has_duplicates:
            raise ValueError("Bars index has duplicate timestamps")
        for c in REQUIRED_COLUMNS:
            df[c] = df[c].astype("float64")
        o, h, l, c, v = (df[k].to_numpy() for k in REQUIRED_COLUMNS)
        if np.any(~np.isfinite(np.c_[o, h, l, c, v])):
            raise ValueError("Bars contain non-finite values")
        body_lo = np.minimum(o, c)
        body_hi = np.maximum(o, c)
        if np.any(l > body_lo + 1e-12) or np.any(h < body_hi - 1e-12):
            raise ValueError("Bars violate low <= min(open,close) <= max(open,close) <= high")
        if np.any(v < 0):
            raise ValueError("Bars contain negative volume")
        if np.any(l <= 0):
            raise ValueError("Bars contain non-positive prices (log-price undefined)")
        self._validated = True
        return self

    # ---- accessors --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    @property
    def ts(self) -> pd.DatetimeIndex:
        return self.df.index

    @property
    def open(self) -> np.ndarray:
        return self.df["open"].to_numpy()

    @property
    def high(self) -> np.ndarray:
        return self.df["high"].to_numpy()

    @property
    def low(self) -> np.ndarray:
        return self.df["low"].to_numpy()

    @property
    def close(self) -> np.ndarray:
        return self.df["close"].to_numpy()

    @property
    def volume(self) -> np.ndarray:
        return self.df["volume"].to_numpy()

    # log-price views (all estimators work in log space)
    @property
    def log_close(self) -> np.ndarray:
        return np.log(self.close)

    @property
    def log_high(self) -> np.ndarray:
        return np.log(self.high)

    @property
    def log_low(self) -> np.ndarray:
        return np.log(self.low)

    def slice(self, start: int, stop: int) -> "Bars":
        """Positional slice returning a new Bars (copy)."""
        return Bars(self.df.iloc[start:stop].copy(), self.symbol, self.interval)


@runtime_checkable
class BarSource(Protocol):
    """The whole data contract. Connectors implement only this."""

    def fetch(self, symbol: str, interval: str, start: datetime, end: datetime) -> Bars: ...
