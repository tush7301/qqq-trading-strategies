"""Contract tests. Every workstream must keep these green.

Tests for not-yet-implemented pieces xfail (NotImplementedError) instead of
failing, so CI stays green while each workstream fills in its part.
"""
import numpy as np
import pandas as pd
import pytest

from qqq import config
from qqq.data import PRICE_COLUMNS, load_prices, split, validate_prices
from qqq.strategies import REGISTRY, validate_positions


def _todo(fn, *a, **k):
    try:
        return fn(*a, **k)
    except NotImplementedError as e:
        pytest.xfail(f"not implemented yet: {e}")


@pytest.fixture(scope="module")
def prices():
    return load_prices(source="synthetic")


def test_synthetic_prices_match_contract(prices):
    validate_prices(prices)
    assert list(prices.columns) == PRICE_COLUMNS


def test_real_prices_match_contract():
    if not config.PRICES_CSV.exists():
        pytest.xfail("WS1: data/raw/QQQ.csv not committed yet")
    validate_prices(load_prices(source="csv"))


def test_split_is_disjoint_and_ordered(prices):
    tr, va, te = split(prices)
    assert tr.index.max() < va.index.min() and va.index.max() < te.index.min()


@pytest.mark.parametrize("name", list(REGISTRY))
def test_strategy_positions_contract(name, prices):
    train, _, _ = split(prices)
    strat = _todo(REGISTRY[name]().fit, train)
    pos = _todo(strat.generate_positions, prices)
    validate_positions(pos, prices)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_strategy_has_no_lookahead(name, prices):
    """Positions up to date T must not change when future data is removed."""
    train, _, _ = split(prices)
    strat = _todo(REGISTRY[name]().fit, train)
    full = _todo(strat.generate_positions, prices)
    cut = prices.index[int(len(prices) * 0.8)]
    trunc = strat.generate_positions(prices.loc[:cut])
    pd.testing.assert_series_equal(full.loc[:cut], trunc, check_names=False)


def test_backtest_timing(prices):
    """Always-long with zero cost must reproduce the asset's returns, lagged correctly."""
    from qqq.backtest import run_backtest

    pos = pd.Series(1.0, index=prices.index)
    res = _todo(run_backtest, prices, pos, cost_bps=0.0)
    asset = prices["AdjClose"].pct_change().fillna(0.0)
    np.testing.assert_allclose(res.returns.iloc[1:], asset.iloc[1:], rtol=1e-12)


def test_metrics_keys(prices):
    from qqq.metrics import METRIC_KEYS, compute_metrics

    r = prices["AdjClose"].pct_change().dropna()
    m = _todo(compute_metrics, r)
    assert set(METRIC_KEYS) <= set(m)
    assert m["max_drawdown"] <= 0
