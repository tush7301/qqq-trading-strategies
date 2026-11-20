"""Data loading (shared infrastructure - done).

Contract: ``load_prices()`` returns a DataFrame
    index   : DatetimeIndex named "Date", trading days, ascending, unique
    columns : Open, High, Low, Close, AdjClose, Volume  (float64)
No NaNs. AdjClose is split/dividend adjusted and is what returns are computed on.

``load_prices(source="synthetic")`` returns fake GBM data with the same shape -
handy for unit tests and for sanity-checking that a strategy has no edge on noise.
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


def download_prices(save: bool = True) -> pd.DataFrame:
    """Download QQQ daily OHLCV from Yahoo Finance and (optionally) save to
    data/raw/QQQ.csv. The committed CSV is the source of truth for the project;
    re-running this may change AdjClose slightly as Yahoo revises dividends.

        python -m qqq.data   # re-download
    """
    import yfinance as yf

    # yfinance's `end` is exclusive -> add one day.
    end = (pd.Timestamp(config.DATA_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(config.TICKER, start=config.DATA_START, end=end,
                      auto_adjust=False, progress=False, multi_level_index=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no data")
    df = raw.rename(columns={"Adj Close": "AdjClose"})[PRICE_COLUMNS].astype("float64")
    df.index = pd.DatetimeIndex(df.index.tz_localize(None) if df.index.tz else df.index, name="Date")
    df = df[~df.index.duplicated()].sort_index().dropna()
    validate_prices(df)
    if save:
        config.DATA_RAW.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.PRICES_CSV, float_format="%.6f")
    return df


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


if __name__ == "__main__":
    d = download_prices()
    print(f"saved {len(d)} rows {d.index[0].date()} -> {d.index[-1].date()} to {config.PRICES_CSV}")
