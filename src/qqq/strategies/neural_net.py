"""Strategy slot 2 - Neural network. Owner: <name>

Idea (from the brief): train a NN on previous days' OHLC data to predict the
next day's move, and trade on the prediction.

Suggested path
  - X = qqq.features.lagged_ohlc_features(prices, lags=self.lags)
    y = (qqq.features.forward_return(prices, 1) > 0)    # target only!
  - Drop warm-up rows and the last row (no target). Fit StandardScaler + model
    on TRAIN rows only - inside fit(). sklearn MLPClassifier is the simplest
    reproducible option (random_state=self.seed); PyTorch LSTM if ambitious.
  - generate_positions: p = predict_proba(X)[:, 1]; position = 1 if p > threshold.
  - Report classification accuracy/AUC on validation AND the trading metrics.
  - Check robustness across 5 seeds; NN results vary a lot by seed.
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
        self.scaler = None

    def fit(self, train_prices: pd.DataFrame) -> "NeuralNetStrategy":
        """TODO: build features/target on train_prices, fit scaler + model."""
        raise NotImplementedError("strategy slot 2: neural network")

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        """TODO: features -> scaler.transform -> predict_proba -> positions (warm-up 0)."""
        raise NotImplementedError("strategy slot 2: neural network")

    def params(self) -> dict:
        return {"lags": self.lags, "hidden": list(self.hidden),
                "threshold": self.threshold, "seed": self.seed}
