"""Strategy registry - add new strategies here so scripts/run_all.py picks them up."""
from qqq.strategies.base import BuyAndHold, Strategy, validate_positions
from qqq.strategies.moving_average import MovingAverageCrossover
from qqq.strategies.neural_net import NeuralNetStrategy

REGISTRY: dict[str, type[Strategy]] = {
    BuyAndHold.name: BuyAndHold,
    MovingAverageCrossover.name: MovingAverageCrossover,
    NeuralNetStrategy.name: NeuralNetStrategy,
}

__all__ = ["Strategy", "BuyAndHold", "MovingAverageCrossover", "NeuralNetStrategy",
           "REGISTRY", "validate_positions"]
