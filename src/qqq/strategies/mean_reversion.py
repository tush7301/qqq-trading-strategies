"""Strategy slot 3 - Mean reversion (technical indicators). Owner: <name>

Idea: short-term overreactions in a broad index tend to partially revert.
Buy when QQQ is oversold, exit when it normalizes.

Suggested path
  - RSI(2..14) entry below `entry` (e.g. 10-30), exit above `exit` (e.g. 50-70);
    or Bollinger %B < 0 entry, > 0.5 exit.
  - Optional trend filter: only go long if AdjClose > SMA(200).
  - Entry/exit with memory = a small state machine: loop over days, or
    build an entry/exit signal Series and forward-fill (pd.Series.ffill).
  - Watch turnover: compare Sharpe at 0 vs 10 bps.
"""
from __future__ import annotations

import pandas as pd

from qqq.strategies.base import Strategy


class MeanReversion(Strategy):
    name = "mean_reversion"

    def __init__(self, rsi_window: int = 2, entry: float = 10.0, exit: float = 70.0,
                 trend_filter: int | None = 200):
        self.rsi_window = rsi_window
        self.entry = entry
        self.exit = exit
        self.trend_filter = trend_filter

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        """TODO."""
        raise NotImplementedError("strategy slot 3: mean reversion")

    def params(self) -> dict:
        return {"rsi_window": self.rsi_window, "entry": self.entry, "exit": self.exit,
                "trend_filter": self.trend_filter}
