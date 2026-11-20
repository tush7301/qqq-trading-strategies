"""WS3 - Moving-average crossover strategy. Owner: <WS3 owner>"""
from __future__ import annotations

import pandas as pd

from qqq.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    name = "ma_crossover"

    def __init__(self, short_window: int = 50, long_window: int = 200, allow_short: bool = False):
        self.short_window = short_window
        self.long_window = long_window
        self.allow_short = allow_short

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        """WS3 TODO: long (1) when SMA_short > SMA_long, else 0 (or -1 if
        allow_short). Use rolling means of AdjClose ending at t (no centering).
        Positions during the long-window warm-up must be 0."""
        raise NotImplementedError("WS3")

    def params(self) -> dict:
        return {"short_window": self.short_window, "long_window": self.long_window,
                "allow_short": self.allow_short}
