"""WS2 - Performance metrics. Owner: <WS2 owner>

compute_metrics(returns) -> dict with EXACTLY these keys (WS5 builds tables from them):
    mean_daily_return, std_daily_return,
    annualized_return      (geometric: (prod(1+r))^(252/n) - 1),
    annualized_mean_return (arithmetic: mean * 252),
    annualized_volatility  (std * sqrt(252)),
    sharpe_ratio           ((mean - rf_daily) / std * sqrt(252)),
    cumulative_return, max_drawdown (negative number), calmar_ratio,
    hit_rate (share of days with r > 0 among days with exposure), n_days
"""
from __future__ import annotations

import pandas as pd

from qqq import config

METRIC_KEYS = [
    "mean_daily_return", "std_daily_return", "annualized_return", "annualized_mean_return",
    "annualized_volatility", "sharpe_ratio", "cumulative_return", "max_drawdown",
    "calmar_ratio", "hit_rate", "n_days",
]


def compute_metrics(
    returns: pd.Series,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
    rf_annual: float = config.RISK_FREE_RATE_ANNUAL,
) -> dict:
    """WS2 TODO."""
    raise NotImplementedError("WS2")


def max_drawdown(equity: pd.Series) -> float:
    """WS2 TODO: min over t of equity_t / running_max(equity)_t - 1."""
    raise NotImplementedError("WS2")
