"""WS4 - Neural-network strategy. Owner: <WS4 owner>

Predict next-day direction (or return) from lagged OHLC features, then map
predictions to positions. Fit ONLY on the training period; scalers too.
"""
from __future__ import annotations

import pandas as pd

from qqq import config
from qqq.strategies.base import Strategy


class NeuralNetStrategy(Strategy):
    name = "neural_net"

    def __init__(self, lags: int = 10, hidden: tuple[int, ...] = (32, 16),
                 threshold: float = 0.5, seed: int = config.RANDOM_SEED):
        self.lags = lags
        self.hidden = hidden
        self.threshold = threshold
        self.seed = seed
        self.model = None

    def fit(self, train_prices: pd.DataFrame) -> "NeuralNetStrategy":
        """WS4 TODO: build features (qqq.features.lagged_ohlc_features, or a
        local copy until WS1 ships), target = 1{r_{t+1} > 0}, fit scaler +
        model on train only, fixed seed."""
        raise NotImplementedError("WS4")

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        """WS4 TODO: p_t = model(features_t); position_t = 1 if p_t > threshold
        else 0 (optionally -1). Warm-up rows -> 0."""
        raise NotImplementedError("WS4")

    def params(self) -> dict:
        return {"lags": self.lags, "hidden": list(self.hidden),
                "threshold": self.threshold, "seed": self.seed}
