"""Data connectors. Adding a source = one module here + one registry line."""
from __future__ import annotations

from ..bars import BarSource


def get_source(name: str, **kwargs) -> BarSource:
    name = name.lower()
    if name == "yfinance":
        from .yfinance_source import YFinanceSource

        return YFinanceSource(**kwargs)
    if name == "csv":
        from .csv_source import CSVSource

        return CSVSource(**kwargs)
    raise KeyError(f"unknown source {name!r}; known: yfinance, csv")


__all__ = ["get_source"]
