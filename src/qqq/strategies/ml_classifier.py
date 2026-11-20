"""Strategy slot 5 - Statistical / classical ML model. Owner: <name>

Idea: a simpler, more interpretable learner than the NN, on a richer feature
set: does it find the same (or any) signal?

Suggested path
  - X = qqq.features.technical_indicators(prices) (+ lagged_ohlc_features)
    y = forward_return(prices, h) > 0   with h in {1, 5}; target only!
  - Models: LogisticRegression (L1/L2) -> RandomForest / GradientBoosting.
    Fit (and any scaler) on TRAIN only. Seed everything.
  - Position = 1 if P(up) > threshold, else 0; or scale by probability.
  - Show feature importance / coefficients: good material for the report.
  - Alternative for this slot: a regime model (e.g. 2-state HMM on returns/vol,
    long only in the calm regime) or ARIMA/GARCH forecasts.
"""
from __future__ import annotations

import pandas as pd

from qqq import config
from qqq.strategies.base import Strategy


class MLClassifierStrategy(Strategy):
    name = "ml_classifier"

    def __init__(self, model: str = "logistic", horizon: int = 1, threshold: float = 0.5,
                 seed: int = config.RANDOM_SEED):
        self.model_name = model
        self.horizon = horizon
        self.threshold = threshold
        self.seed = seed
        self.model = None

    def fit(self, train_prices: pd.DataFrame) -> "MLClassifierStrategy":
        """TODO."""
        raise NotImplementedError("strategy slot 5: ML classifier")

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        """TODO."""
        raise NotImplementedError("strategy slot 5: ML classifier")

    def params(self) -> dict:
        return {"model": self.model_name, "horizon": self.horizon,
                "threshold": self.threshold, "seed": self.seed}
