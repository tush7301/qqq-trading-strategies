"""WS2 - Backtest engine. Owner: <WS2 owner>

Contract:
    run_backtest(prices, positions, cost_bps) -> BacktestResult
    BacktestResult.returns : pd.Series of daily net strategy returns (index = prices.index,
                             first row 0)
    BacktestResult.equity  : pd.Series, cumulative growth of $1 = (1 + returns).cumprod()
    BacktestResult.turnover: pd.Series, |position_t - position_{t-1}|

Timing (see strategies/base.py):
    gross_t = positions_{t-1} * asset_ret_t
    cost_t  = turnover_{t-1} * cost_bps / 1e4      (traded at close t-1)
    net_t   = gross_t - cost_t
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qqq import config


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    turnover: pd.Series
    positions: pd.Series


def run_backtest(
    prices: pd.DataFrame,
    positions: pd.Series,
    cost_bps: float = config.TRANSACTION_COST_BPS,
) -> BacktestResult:
    """WS2 TODO: implement per the timing rules in the module docstring.
    Call qqq.strategies.validate_positions first."""
    raise NotImplementedError("WS2")
