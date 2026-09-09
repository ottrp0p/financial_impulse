"""Price and forecast store: parquet per symbol, delta fetch, incremental rolling fits.

prices/{SYM}.parquet     : open high low close volume, UTC ts index
forecasts/{SYM}.parquet  : one row per (issue_pos, h): the band the field issued at that close
                           columns: issue_pos, issue_date, h, x0, m_q10, m_q50, m_q90, b_q10, b_q50, b_q90
issue_pos is the positional index of the close the forecast was issued on (x0 = log close there).
The band *targets* position issue_pos + h.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from impulse.bars import Bars
from impulse.forecast import brownian_samples, model_samples
from impulse.grid import PriceGrid
from impulse.potential import fit_field
from impulse.sources import get_source

FIELD = {"fit_bars": 120, "h": 10, "shrink_n0": 50.0, "sigma_mode": "bar", "n_paths": 2000, "min_n_eff": 10.0}
HISTORY_DAYS = 600          # calendar days of price history to keep
FIELD_VERSION = "f1"        # bump when FIELD changes -> forecasts are refit
ET = ZoneInfo("America/New_York")
CLOSE_HHMM = (16, 5)        # a session's bar is only trusted after this ET time


def session_open_now() -> tuple[bool, str]:
    """(True, today's ET date) while today's US session bar is still forming; (False, date) after the close."""
    now = datetime.now(ET)
    d = now.strftime("%Y-%m-%d")
    before_close = (now.hour, now.minute) < CLOSE_HHMM
    return (now.weekday() < 5 and before_close), d


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "prices").mkdir(parents=True, exist_ok=True)
        (self.root / "forecasts").mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._glock = threading.Lock()
        self.progress: dict[str, str] = {}

    def _lock(self, sym: str) -> threading.Lock:
        with self._glock:
            return self._locks.setdefault(sym, threading.Lock())

    # ---- prices -----------------------------------------------------------
    def price_path(self, sym: str) -> Path:
        return self.root / "prices" / f"{sym}.parquet"

    def load_prices(self, sym: str) -> pd.DataFrame | None:
        p = self.price_path(sym)
        return pd.read_parquet(p) if p.exists() else None

    def refresh_prices(self, sym: str, end: datetime | None = None) -> pd.DataFrame:
        """Delta fetch: only pull bars after the last stored date (with a 5-day overlap for revisions)."""
        end = end or datetime.utcnow() + timedelta(days=1)
        old = self.load_prices(sym)
        src = get_source("yfinance", use_cache=False)
        if old is None or len(old) == 0:
            start = end - timedelta(days=HISTORY_DAYS)
        else:
            start = old.index[-1].to_pydatetime().replace(tzinfo=None) - timedelta(days=5)
        try:
            new = src.fetch(sym, "1d", start, end).df
        except Exception as e:
            if old is None:
                raise
            return old
        df = new if old is None else pd.concat([old[old.index < new.index[0]], new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # never store the current session's bar while it is still forming (yfinance returns a partial bar intraday)
        forming, today_et = session_open_now()
        if forming:
            df = df[df.index.strftime("%Y-%m-%d") < today_et]
        df = df[df.index >= df.index[-1] - pd.Timedelta(days=HISTORY_DAYS)]
        df.to_parquet(self.price_path(sym))
        return df

    # ---- forecasts --------------------------------------------------------
    def fc_path(self, sym: str) -> Path:
        return self.root / "forecasts" / f"{sym}.{FIELD_VERSION}.parquet"

    def load_forecasts(self, sym: str) -> pd.DataFrame | None:
        p = self.fc_path(sym)
        return pd.read_parquet(p) if p.exists() else None

    def refresh_forecasts(self, sym: str, prices: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        """Fit the field at every close that doesn't have a forecast yet; append."""
        bars = Bars(prices.copy(), sym, "1d")
        x = bars.log_close
        N, h = FIELD["fit_bars"], FIELD["h"]
        old = self.load_forecasts(sym)
        # positions are keyed by date so a history trim doesn't shift them: store issue_date, recompute pos
        done = set(pd.to_datetime(old["issue_date"]).dt.strftime("%Y-%m-%d")) if old is not None and len(old) else set()
        dates = [d.strftime("%Y-%m-%d") for d in bars.ts]
        todo = [i for i in range(N, len(bars)) if dates[i - 1] not in done]   # forecast issued at close i-1 using bars [i-N, i)
        rows = []
        rng = np.random.default_rng(seed)
        for k, t in enumerate(todo):
            if k % 20 == 0:
                self.progress[sym] = f"fitting {k}/{len(todo)}"
            fit = bars.slice(t - N, t)
            grid = PriceGrid.from_bars(fit)
            fld = fit_field(fit, grid, tau=N / 2, min_n_eff=FIELD["min_n_eff"])
            sigma_bar = float(np.std(np.diff(fit.log_close)))
            x0 = float(x[t - 1])
            sm = model_samples(fld, x0, h, sigma_bar, FIELD["n_paths"], rng, shrink_n0=FIELD["shrink_n0"], sigma_mode=FIELD["sigma_mode"])
            sb = brownian_samples(x0, h, sigma_bar, FIELD["n_paths"], rng)
            qm, qb = np.quantile(sm, [0.1, 0.5, 0.9]), np.quantile(sb, [0.1, 0.5, 0.9])
            rows.append({"issue_date": dates[t - 1], "h": h, "x0": x0, "m_q10": qm[0], "m_q50": qm[1], "m_q90": qm[2],
                         "b_q10": qb[0], "b_q50": x0, "b_q90": qb[2]})
        self.progress.pop(sym, None)
        new = pd.DataFrame(rows)
        df = new if old is None or len(old) == 0 else pd.concat([old, new], ignore_index=True)
        if len(df):
            df = df.drop_duplicates("issue_date", keep="last").sort_values("issue_date").reset_index(drop=True)
            df.to_parquet(self.fc_path(sym))
        return df

    def ensure(self, sym: str, refetch: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prices + forecasts for a symbol, refreshed. Thread-safe per symbol."""
        with self._lock(sym):
            prices = self.refresh_prices(sym) if refetch else self.load_prices(sym)
            if prices is None:
                prices = self.refresh_prices(sym)
            if len(prices) < FIELD["fit_bars"] + 30:
                raise ValueError(f"{sym}: only {len(prices)} bars of history; need at least {FIELD['fit_bars'] + 30}")
            fc = self.refresh_forecasts(sym, prices)
            return prices, fc

    def status(self, sym: str) -> dict:
        p = self.load_prices(sym); f = self.load_forecasts(sym)
        return {"symbol": sym, "bars": 0 if p is None else len(p), "last_bar": None if p is None or not len(p) else p.index[-1].strftime("%Y-%m-%d"),
                "forecasts": 0 if f is None else len(f), "progress": self.progress.get(sym)}
