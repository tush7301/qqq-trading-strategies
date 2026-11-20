"""Data loading (shared infrastructure - done).

Contract: ``load_prices()`` returns a DataFrame
    index   : DatetimeIndex named "Date", trading days, ascending, unique
    columns : Open, High, Low, Close, AdjClose, Volume  (float64)
No NaNs. AdjClose is split/dividend adjusted and is what returns are computed on.

``load_prices(source="synthetic")`` returns fake GBM data with the same shape -
handy for unit tests and for sanity-checking that a strategy has no edge on noise.

``risk_free_daily(index)`` returns the daily T-bill return EARNED on each day
(cash held from close t-1 to close t earns the rate known at close t-1).
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



# ------------------------------------------------------------ risk-free rate ---
def download_risk_free(save: bool = True) -> pd.Series:
    """13-week T-bill yield (percent, annualized) from Yahoo (^IRX)."""
    import yfinance as yf

    start = (pd.Timestamp(config.DATA_START) - pd.Timedelta(days=31)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(config.DATA_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(config.RISK_FREE_TICKER, start=start, end=end, auto_adjust=False,
                      progress=False, multi_level_index=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no T-bill data")
    s = raw["Close"].astype("float64").rename("irx_pct")
    s.index = pd.DatetimeIndex(s.index.tz_localize(None) if s.index.tz else s.index, name="Date")
    s = s[~s.index.duplicated()].sort_index().dropna()
    if save:
        s.to_frame().to_csv(config.RISK_FREE_CSV, float_format="%.4f")
    return s


def load_risk_free_yield() -> pd.Series:
    """Annualized 13-week T-bill yield in percent, as committed in data/raw/IRX.csv."""
    return pd.read_csv(config.RISK_FREE_CSV, parse_dates=["Date"], index_col="Date")["irx_pct"]


def risk_free_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Daily T-bill return earned ON each date in `index`.

    rf_t = (1 + y_{t-1}/100)^(1/252) - 1, where y_{t-1} is the yield known at the
    previous close (forward-filled over holidays). First day -> 0. Dates outside the
    committed data (e.g. synthetic tests) are filled with the nearest known yield,
    or 0 if the file is missing.
    """
    if not config.RISK_FREE_CSV.exists():
        return pd.Series(0.0, index=index, name="rf")
    y = load_risk_free_yield()
    y = y.reindex(y.index.union(index)).ffill().bfill().reindex(index)
    daily = (1 + y / 100) ** (1 / config.TRADING_DAYS_PER_YEAR) - 1
    return daily.shift(1).fillna(0.0).rename("rf")

if __name__ == "__main__":
    d = download_prices()
    print(f"saved {len(d)} rows {d.index[0].date()} -> {d.index[-1].date()} to {config.PRICES_CSV}")
    r = download_risk_free()
    print(f"saved {len(r)} rows {r.index[0].date()} -> {r.index[-1].date()} to {config.RISK_FREE_CSV}")
