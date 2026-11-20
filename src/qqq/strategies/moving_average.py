"""Strategy slot 1 - Moving-average crossover. Owner: <name>

Idea: long when the short-term average is above the long-term average
(trend up), flat (or short) otherwise.

Suggested path
  - SMA crossover on AdjClose; then try EMA, and a +/- band to cut whipsaw.
  - grid_search(MovingAverageCrossover, {"short_window": [5, 10, 20, 50],
                                         "long_window": [50, 100, 150, 200]})
  - Pick the final pair on VALIDATION; set it as the default below.
"""
from __future__ import annotations

import pandas as pd

from qqq.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    name = "ma_crossover"

    def __init__(self, short_window: int = 50, long_window: int = 200, allow_short: bool = False):
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.allow_short = allow_short

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        """TODO: 1 when SMA_short > SMA_long, else 0 (-1 if allow_short).
        Warm-up (first long_window-1 days) must be 0."""
        raise NotImplementedError("strategy slot 1: moving average")

    def params(self) -> dict:
        return {"short_window": self.short_window, "long_window": self.long_window,
                "allow_short": self.allow_short}
