"""WS1 - Data. Owner: <WS1 owner>

Contract: ``load_prices()`` returns a DataFrame
    index   : DatetimeIndex named "Date", trading days, ascending, unique
    columns : Open, High, Low, Close, AdjClose, Volume  (float64)
No NaNs. AdjClose is split/dividend adjusted and is what returns are computed on.

Until WS1 commits ``data/raw/QQQ.csv``, everyone can develop against
``load_prices(source="synthetic")``, which has the exact same shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qqq import config

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]


def load_prices(
    source: str = "auto",
    start: str = config.DATA_START,
    end: str = config.DATA_END,
) -> pd.DataFrame:
    """Load daily QQQ OHLCV.

    source: "csv" (committed file), "synthetic" (fake GBM data for dev/tests),
            or "auto" (csv if present, else synthetic with a warning).
    """
    if source == "auto":
        source = "csv" if config.PRICES_CSV.exists() else "synthetic"
        if source == "synthetic":
            import warnings

            warnings.warn("data/raw/QQQ.csv not found - using SYNTHETIC prices.")
    if source == "csv":
        df = pd.read_csv(config.PRICES_CSV, parse_dates=["Date"], index_col="Date")
    elif source == "synthetic":
        df = synthetic_prices(start, end)
    else:
        raise ValueError(f"unknown source {source!r}")
    df = df.loc[start:end, PRICE_COLUMNS].astype("float64")
    validate_prices(df)
    return df


def download_prices() -> pd.DataFrame:
    """WS1 TODO: download QQQ daily OHLCV (e.g. yfinance, auto_adjust=False),
    rename 'Adj Close' -> 'AdjClose', clean, save to config.PRICES_CSV.
    Document source + download date in data/README.md."""
    raise NotImplementedError("WS1")


def split(df: pd.DataFrame | pd.Series):
    """Return (train, validation, test) slices using config dates."""
    return (
        df.loc[: config.TRAIN_END],
        df.loc[config.VALIDATION_START : config.VALIDATION_END],
        df.loc[config.TEST_START :],
    )


def validate_prices(df: pd.DataFrame) -> None:
    assert list(df.columns) == PRICE_COLUMNS, df.columns
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.name == "Date"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert not df.isna().any().any(), "NaNs in price data"
    assert (df[["Open", "High", "Low", "Close", "AdjClose"]] > 0).all().all()


def synthetic_prices(start: str, end: str, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Geometric-Brownian-motion OHLCV with the real contract's shape."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end, name="Date")
    r = rng.normal(0.0004, 0.013, len(idx))
    close = 40 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.004, len(idx)))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.005, len(idx))))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.005, len(idx))))
    vol = rng.lognormal(17, 0.3, len(idx))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "AdjClose": close, "Volume": vol},
        index=idx,
    )
