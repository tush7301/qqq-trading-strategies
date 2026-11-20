"""Shared feature library (infrastructure - done; strategy owners may add more).

RULE: a feature value at date t may only use data up to and including the
CLOSE of day t. Every function here is "causal" (rolling windows end at t,
nothing is centered, nothing is normalized with full-sample statistics).
tests/test_features.py checks this by truncating the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- returns ---
def daily_returns(prices: pd.DataFrame) -> pd.Series:
    """Simple close-to-close returns on AdjClose: r_t = P_t / P_{t-1} - 1."""
    return prices["AdjClose"].pct_change().rename("ret")


def log_returns(prices: pd.DataFrame) -> pd.Series:
    return np.log(prices["AdjClose"]).diff().rename("logret")


def forward_return(prices: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """TARGET ONLY - never use as a feature. r_{t+1..t+h} aligned at t."""
    p = prices["AdjClose"]
    return (p.shift(-horizon) / p - 1).rename(f"fwd_ret_{horizon}")


# ----------------------------------------------------------- moving avgs ---
def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


# -------------------------------------------------------------- oscillators ---
def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]."""
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = up / dn
    return (100 - 100 / (1 + rs)).rename(f"rsi_{window}")


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger_pct_b(s: pd.Series, window: int = 20, k: float = 2.0) -> pd.Series:
    """%B: 0 = lower band, 1 = upper band."""
    m = sma(s, window)
    sd = s.rolling(window, min_periods=window).std()
    return ((s - (m - k * sd)) / (2 * k * sd)).rename(f"pctb_{window}")


def rolling_volatility(prices: pd.DataFrame, window: int = 20, annualize: bool = True) -> pd.Series:
    v = daily_returns(prices).rolling(window, min_periods=window).std()
    return (v * np.sqrt(252) if annualize else v).rename(f"vol_{window}")


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 0) -> pd.Series:
    """Total return over [t-lookback, t-skip]."""
    p = prices["AdjClose"]
    return (p.shift(skip) / p.shift(lookback) - 1).rename(f"mom_{lookback}_{skip}")


def zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling (past-only) z-score."""
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std()
    return (s - m) / sd


# ------------------------------------------------------------ feature sets ---
def lagged_ohlc_features(prices: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Scale-free daily features, lagged 0..lags-1 (lag 0 = today's close).

    Per day: log return, intraday range (H-L)/C, body (C-O)/O, overnight gap
    O_t/C_{t-1}-1, and 20-day volume z-score. Rows with NaN = warm-up.
    """
    adj = prices["AdjClose"] / prices["Close"]  # adjust OHLC consistently with AdjClose
    o, h, l, c = (prices[k] * adj for k in ("Open", "High", "Low", "Close"))
    base = pd.DataFrame(
        {
            "logret": np.log(c).diff(),
            "range": (h - l) / c,
            "body": (c - o) / o,
            "gap": o / c.shift(1) - 1,
            "vol_z": zscore(np.log(prices["Volume"]), 20),
        },
        index=prices.index,
    )
    cols = {f"{k}_lag{i}": base[k].shift(i) for i in range(lags) for k in base}
    return pd.DataFrame(cols, index=prices.index)


def technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """A standard indicator panel, all scale-free (ratios, not price levels)."""
    p = prices["AdjClose"]
    out = pd.DataFrame(index=prices.index)
    for w in (10, 20, 50, 100, 200):
        out[f"px_sma{w}"] = p / sma(p, w) - 1
    out["sma50_sma200"] = sma(p, 50) / sma(p, 200) - 1
    out["rsi_14"] = rsi(p, 14)
    out = out.join(macd(p).div(p, axis=0))
    out["pctb_20"] = bollinger_pct_b(p, 20)
    out["vol_20"] = rolling_volatility(prices, 20)
    out["vol_60"] = rolling_volatility(prices, 60)
    out["mom_21"] = momentum(prices, 21)
    out["mom_63"] = momentum(prices, 63)
    out["mom_252_21"] = momentum(prices, 252, 21)
    out["drawdown_252"] = p / p.rolling(252, min_periods=1).max() - 1
    return out
