"""COPY ME to start a new strategy:  cp _template.py my_strategy.py

Checklist
  1. Rename the class and set a unique `name` (snake_case; used for file names).
  2. Give every hyper-parameter a default in __init__ - run_all.py calls `Cls()`.
     Once you pick final params on VALIDATION, make them the defaults.
  3. fit(): learn anything data-driven (models, scalers, thresholds) from
     `train_prices` ONLY. Rule-based strategies can leave it as a no-op.
  4. generate_positions(): return a Series indexed like `prices`, values in
     [-1, 1], where position[t] uses data up to the close of t. Do NOT shift -
     the engine does. Fill warm-up NaNs with 0.
  5. Register it in strategies/__init__.py and run `pytest` - the contract
     and no-look-ahead tests pick it up automatically.
  6. Notebook loop:  evaluate(MyStrategy()), grid_search(MyStrategy, {...}).

Useful building blocks: qqq.features (sma, ema, rsi, macd, bollinger_pct_b,
rolling_volatility, momentum, zscore, lagged_ohlc_features, technical_indicators,
forward_return [targets only]).
"""
from __future__ import annotations

import pandas as pd

from qqq import features as F
from qqq.strategies.base import Strategy


class MyStrategy(Strategy):
    name = "my_strategy"

    def __init__(self, window: int = 20):
        self.window = window

    def fit(self, train_prices: pd.DataFrame) -> "MyStrategy":
        return self

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        signal = prices["AdjClose"] > F.sma(prices["AdjClose"], self.window)
        return signal.astype(float).rename("position")  # warm-up: NaN>x is False -> 0

    def params(self) -> dict:
        return {"window": self.window}
