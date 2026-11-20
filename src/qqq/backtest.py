"""Backtest engine (infrastructure - done).

    run_backtest(prices, positions, cost_bps) -> BacktestResult

Timing (see strategies/base.py):
    position_t is chosen with information up to the close of t and is held
    from close t to close t+1. The engine - and ONLY the engine - lags it:

    gross_t  = position_{t-1} * asset_ret_t
    cost_t   = |position_{t-1} - position_{t-2}| * cost_bps / 1e4
               (the trade happened at close t-1; its cost is booked on day t)
    cash_t   = (1 - |position_{t-1}|) * rf_t        (uninvested capital earns T-bills)
    net_t    = gross_t - cost_t + cash_t

rf_t is the daily 13-week T-bill return earned on day t (qqq.data.risk_free_daily).
Short positions earn -asset_ret; we conservatively assume short-sale proceeds earn
nothing and ignore borrow fees (state in report).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qqq import config
from qqq.features import daily_returns


@dataclass
class BacktestResult:
    returns: pd.Series    # daily net strategy returns, first row 0
    gross: pd.Series      # before costs
    equity: pd.Series     # growth of $1
    turnover: pd.Series   # |pos_t - pos_{t-1}| traded at close t
    positions: pd.Series  # as given by the strategy (un-lagged)
    rf: pd.Series         # daily T-bill return earned on each day (0 if disabled)

    def slice(self, start=None, end=None) -> "BacktestResult":
        """Sub-period view with equity re-based to 1 at the start."""
        r = self.returns.loc[start:end]
        return BacktestResult(
            returns=r,
            gross=self.gross.loc[start:end],
            equity=(1 + r).cumprod(),
            turnover=self.turnover.loc[start:end],
            positions=self.positions.loc[start:end],
            rf=self.rf.loc[start:end],
        )

    def metrics(self) -> dict:
        """compute_metrics (Sharpe/Sortino in excess of T-bills) + trade_stats."""
        from qqq.metrics import compute_metrics

        return compute_metrics(self.returns, rf_daily=self.rf) | trade_stats(self)


def run_backtest(
    prices: pd.DataFrame,
    positions: pd.Series,
    cost_bps: float = config.TRANSACTION_COST_BPS,
    cash_rate: pd.Series | str | None = "auto",
) -> BacktestResult:
    """cash_rate: "auto" = committed T-bill series if config.CASH_EARNS_RISK_FREE,
    a daily-rate Series (earned on day t), or None for 0% on cash."""
    from qqq.data import risk_free_daily
    from qqq.strategies.base import validate_positions

    validate_positions(positions, prices)
    pos = positions.astype("float64")
    asset = daily_returns(prices).fillna(0.0)

    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])  # entering from cash

    if isinstance(cash_rate, str):
        if cash_rate != "auto":
            raise ValueError("cash_rate must be 'auto', a Series or None")
        rf = risk_free_daily(prices.index) if config.CASH_EARNS_RISK_FREE else None
    else:
        rf = cash_rate
    rf = pd.Series(0.0, index=prices.index) if rf is None else rf.reindex(prices.index).fillna(0.0)

    held = pos.shift(1).fillna(0.0)
    gross = held * asset
    cost = turnover.shift(1).fillna(0.0) * cost_bps / 1e4
    cash = (1.0 - held.abs()) * rf
    net = (gross - cost + cash).rename("strategy_ret")

    return BacktestResult(
        returns=net,
        gross=gross.rename("gross_ret"),
        equity=(1 + net).cumprod().rename("equity"),
        turnover=turnover.rename("turnover"),
        positions=pos.rename("position"),
        rf=rf.rename("rf"),
    )


def trade_stats(result: BacktestResult) -> dict:
    """Activity statistics for a (possibly sliced) backtest."""
    pos, to = result.positions, result.turnover
    changes = int((to > 1e-12).sum())
    n = len(pos)
    exposure = float(pos.abs().mean()) if n else np.nan
    return {
        "n_position_changes": changes,
        "avg_holding_days": float(n / changes) if changes else float(n),
        "exposure": exposure,
        "pct_days_long": float((pos > 0).mean()) if n else np.nan,
        "pct_days_short": float((pos < 0).mean()) if n else np.nan,
        "annual_turnover": float(to.sum() * config.TRADING_DAYS_PER_YEAR / n) if n else np.nan,
    }
