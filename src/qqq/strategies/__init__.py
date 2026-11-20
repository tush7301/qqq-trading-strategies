"""Strategy registry - scripts/run_all.py and the tests iterate over REGISTRY.

To add a strategy: copy _template.py, implement it, add one line below.
"""
from qqq.strategies.base import BuyAndHold, Strategy, validate_positions
from qqq.strategies.mean_reversion import MeanReversion
from qqq.strategies.ml_classifier import MLClassifierStrategy
from qqq.strategies.momentum import MomentumVolTarget
from qqq.strategies.moving_average import MovingAverageCrossover
from qqq.strategies.neural_net import NeuralNetStrategy

REGISTRY: dict[str, type[Strategy]] = {
    BuyAndHold.name: BuyAndHold,                         # benchmark
    MovingAverageCrossover.name: MovingAverageCrossover,  # slot 1
    NeuralNetStrategy.name: NeuralNetStrategy,            # slot 2
    MeanReversion.name: MeanReversion,                    # slot 3
    MomentumVolTarget.name: MomentumVolTarget,            # slot 4
    MLClassifierStrategy.name: MLClassifierStrategy,      # slot 5
}

__all__ = ["Strategy", "BuyAndHold", "validate_positions", "REGISTRY",
           "MovingAverageCrossover", "NeuralNetStrategy", "MeanReversion",
           "MomentumVolTarget", "MLClassifierStrategy"]
