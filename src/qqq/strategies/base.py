"""Strategy contract - FROZEN. Every strategy (WS3, WS4, ...) implements this.

Timing convention (the single most important rule for avoiding look-ahead):
    positions[t] is decided using information available at the CLOSE of day t
    and is held from close t to close t+1. The backtest engine (WS2) does the
    shift: strategy_return[t+1] = positions[t] * asset_return[t+1] - costs.
    Strategies must NOT shift positions themselves.

positions: pd.Series, same index as the prices passed in, float in [-1, 1]
    (1 = fully long, 0 = flat, -1 = fully short). Warm-up periods -> 0.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"

    def fit(self, train_prices: pd.DataFrame) -> "Strategy":
        """Fit parameters / model on the training period only. Default: no-op."""
        return self

    @abstractmethod
    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        """Return positions per the contract above."""

    def params(self) -> dict:
        """Hyper-parameters, for logging in results/metrics."""
        return {}


class BuyAndHold(Strategy):
    """Benchmark required by the brief."""

    name = "buy_and_hold"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=prices.index, name="position")


def validate_positions(positions: pd.Series, prices: pd.DataFrame) -> None:
    assert isinstance(positions, pd.Series), "positions must be a pd.Series"
    assert positions.index.equals(prices.index), "positions index must equal prices index"
    assert not positions.isna().any(), "NaN positions - fill warm-up with 0"
    assert positions.between(-1, 1).all(), "positions must lie in [-1, 1]"
