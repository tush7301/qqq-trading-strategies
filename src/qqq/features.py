"""WS1 - Shared features. Owner: <WS1 owner>

Rule: the feature value at date t may use data up to and including the CLOSE
of t, never t+1. (The engine applies the position from t's close onward.)
"""
from __future__ import annotations

import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.Series:
    """Simple close-to-close returns on AdjClose. r_t = P_t / P_{t-1} - 1."""
    return prices["AdjClose"].pct_change().rename("ret")


def lagged_ohlc_features(prices: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """WS1 TODO: lagged, scale-free OHLC features for WS4 (e.g. log returns,
    (High-Low)/Close, (Close-Open)/Open, volume z-score) for lags 0..lags-1."""
    raise NotImplementedError("WS1")


def technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """WS1 TODO (stretch): SMA/EMA, RSI, MACD, Bollinger %B, rolling vol."""
    raise NotImplementedError("WS1")
