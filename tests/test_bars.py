import numpy as np
import pandas as pd
import pytest

from impulse.bars import Bars, BarSource
from impulse.sources import get_source


def _df(n=5):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = np.linspace(100, 104, n)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.ones(n)}, index=idx)


def test_bars_validates_and_localises_utc():
    b = Bars(_df(), "X", "1h")
    assert str(b.ts.tz) == "UTC" and b.ts.name == "ts" and len(b) == 5
    assert np.allclose(b.log_close, np.log(b.close))


def test_bars_rejects_bad_ohlc():
    df = _df()
    df.loc[df.index[2], "high"] = 50
    with pytest.raises(ValueError):
        Bars(df)


def test_csv_source_is_a_barsource(tmp_path):
    df = _df().rename_axis("date")
    (tmp_path / "X_1h.csv").write_text(df.to_csv())
    src = get_source("csv", root=tmp_path)
    assert isinstance(src, BarSource)
    b = src.fetch("X", "1h", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
    assert len(b) == 5 and b.symbol == "X"
