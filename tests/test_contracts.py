"""Strategy contract tests - run automatically for EVERY registered strategy.

Unimplemented strategy slots xfail (NotImplementedError) so CI stays green
while each owner fills in their strategy.
"""
import pandas as pd
import pytest

from qqq import config
from qqq.data import PRICE_COLUMNS, load_prices, split, validate_prices
from qqq.evaluate import check_no_lookahead
from qqq.strategies import REGISTRY, validate_positions


def _todo(fn, *a, **k):
    try:
        return fn(*a, **k)
    except NotImplementedError as e:
        pytest.xfail(f"not implemented yet: {e}")


@pytest.fixture(scope="module")
def prices():
    return load_prices(source="csv")


def test_committed_data_matches_contract(prices):
    assert config.PRICES_CSV.exists()
    validate_prices(prices)
    assert list(prices.columns) == PRICE_COLUMNS
    assert prices.index[0] >= pd.Timestamp(config.DATA_START)
    assert prices.index[-1] <= pd.Timestamp(config.DATA_END)
    assert len(prices) > 5000


def test_committed_tbill_data():
    from qqq.data import load_risk_free_yield
    y = load_risk_free_yield()
    assert config.RISK_FREE_CSV.exists() and y.notna().all()
    assert y.index[0] <= pd.Timestamp(config.DATA_START)
    assert y.index[-1] >= pd.Timestamp(config.DATA_END) - pd.Timedelta(days=7)
    assert y.between(-0.5, 20).all()


def test_synthetic_matches_contract():
    validate_prices(load_prices(source="synthetic"))


def test_split_is_disjoint_and_ordered(prices):
    tr, va, te = split(prices)
    assert len(tr) and len(va) and len(te)
    assert tr.index.max() < va.index.min() and va.index.max() < te.index.min()
    assert len(tr) + len(va) + len(te) == len(prices)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_strategy_positions_contract(name, prices):
    train, _, _ = split(prices)
    strat = _todo(REGISTRY[name]().fit, train)
    pos = _todo(strat.generate_positions, prices)
    validate_positions(pos, prices)
    assert isinstance(strat.params(), dict)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_strategy_has_no_lookahead(name, prices):
    """Positions up to date T must not change when data after T is removed."""
    _todo(check_no_lookahead, REGISTRY[name](), prices)
