"""One-stop helpers for strategy owners. Typical notebook usage:

    from qqq.evaluate import evaluate, grid_search, check_no_lookahead
    from qqq.strategies.mean_reversion import MeanReversion

    evaluate(MeanReversion(window=10))                       # train + validation table
    grid_search(MeanReversion, {"window": [5, 10, 20]})      # ranked on TRAIN Sharpe
    check_no_lookahead(MeanReversion())                      # raises if it peeks

Never pass period="test" while developing - the test set is run once, at the end.
"""
from __future__ import annotations

import itertools
import warnings

import pandas as pd

from qqq import config
from qqq.backtest import run_backtest
from qqq.data import load_prices, split
from qqq.strategies.base import BuyAndHold, Strategy, validate_positions

PERIODS = {
    "train": (None, config.TRAIN_END),
    "validation": (config.VALIDATION_START, config.VALIDATION_END),
    "test": (config.TEST_START, None),
}
DEV_PERIODS = ("train", "validation")


def backtest_strategy(strategy: Strategy, prices: pd.DataFrame | None = None,
                      cost_bps: float = config.TRANSACTION_COST_BPS):
    """Fit on train, generate positions on the full sample, run the engine."""
    prices = load_prices() if prices is None else prices
    train, _, _ = split(prices)
    strategy.fit(train)
    pos = strategy.generate_positions(prices)
    validate_positions(pos, prices)
    return run_backtest(prices, pos, cost_bps=cost_bps)


def period_metrics(result, periods=DEV_PERIODS) -> pd.DataFrame:
    rows = {}
    for p in periods:
        rows[p] = result.slice(*PERIODS[p]).metrics()
    return pd.DataFrame(rows).T


def evaluate(strategy: Strategy, prices: pd.DataFrame | None = None,
             periods=DEV_PERIODS, cost_bps: float = config.TRANSACTION_COST_BPS,
             with_benchmark: bool = True) -> pd.DataFrame:
    """Metrics per period for a strategy (and buy-and-hold for reference)."""
    if "test" in periods:
        warnings.warn("Evaluating on the TEST period. Only do this at final collation.")
    prices = load_prices() if prices is None else prices
    frames = {strategy.name: period_metrics(backtest_strategy(strategy, prices, cost_bps), periods)}
    if with_benchmark:
        frames["buy_and_hold"] = period_metrics(backtest_strategy(BuyAndHold(), prices, cost_bps), periods)
    return pd.concat(frames, names=["strategy", "period"])


def grid_search(strategy_cls: type[Strategy], grid: dict, period: str = "train",
                metric: str = "sharpe_ratio", prices: pd.DataFrame | None = None,
                cost_bps: float = config.TRANSACTION_COST_BPS, **fixed) -> pd.DataFrame:
    """Evaluate every parameter combination; returns a table sorted by `metric`.
    Rank on TRAIN, then confirm the top few on VALIDATION (period='validation')."""
    if period == "test":
        raise ValueError("grid_search on the test period is not allowed.")
    prices = load_prices() if prices is None else prices
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo)) | fixed
        try:
            res = backtest_strategy(strategy_cls(**params), prices, cost_bps)
        except ValueError as e:  # e.g. short window >= long window
            warnings.warn(f"skipped {params}: {e}")
            continue
        m = res.slice(*PERIODS[period]).metrics()
        rows.append(dict(zip(keys, combo)) | m)
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


def check_no_lookahead(strategy: Strategy, prices: pd.DataFrame | None = None,
                       cut_fracs=(0.5, 0.7, 0.9)) -> None:
    """Positions up to date T must be identical whether or not data after T exists."""
    prices = load_prices() if prices is None else prices
    train, _, _ = split(prices)
    strategy.fit(train)
    full = strategy.generate_positions(prices)
    for f in cut_fracs:
        cut = prices.index[int(len(prices) * f)]
        trunc = strategy.generate_positions(prices.loc[:cut])
        pd.testing.assert_series_equal(full.loc[:cut], trunc, check_names=False,
                                       obj=f"positions truncated at {cut.date()}")
