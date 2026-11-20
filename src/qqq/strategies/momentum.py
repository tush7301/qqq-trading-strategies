"""Strategy slot 4 - Time-series momentum / volatility targeting. Owner: <name>

Idea: positive past returns predict positive future returns (trend persistence),
and scaling exposure inversely to recent volatility improves risk-adjusted returns
(volatility clusters; crashes happen in high-vol regimes).

Suggested path
  - Signal: sign of the trailing `lookback`-day return (e.g. 63/126/252, optionally
    skipping the most recent month).
  - Sizing: position = signal * min(1, target_vol / realized_vol_20d)  -> in [0, 1]
    (keeps us within the [-1, 1] contract; no leverage).
  - Compare: momentum only, vol-targeting only (always long, scaled), both.
"""
from __future__ import annotations

import pandas as pd

from qqq.strategies.base import Strategy


class MomentumVolTarget(Strategy):
    name = "momentum_vol_target"

    def __init__(self, lookback: int = 126, skip: int = 0, target_vol: float | None = 0.15,
                 vol_window: int = 20, allow_short: bool = False):
        self.lookback = lookback
        self.skip = skip
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.allow_short = allow_short

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        """TODO."""
        raise NotImplementedError("strategy slot 4: momentum / vol targeting")

    def params(self) -> dict:
        return {"lookback": self.lookback, "skip": self.skip, "target_vol": self.target_vol,
                "vol_window": self.vol_window, "allow_short": self.allow_short}
