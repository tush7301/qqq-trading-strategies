"""Unit tests for the backtest engine, metrics and features."""
import numpy as np
import pandas as pd
import pytest

from qqq import features as F
from qqq.backtest import run_backtest, trade_stats
from qqq.data import load_prices
from qqq.metrics import METRIC_KEYS, compute_metrics, max_drawdown


@pytest.fixture(scope="module")
def prices():
    return load_prices(source="csv")


def _toy():
    idx = pd.bdate_range("2020-01-01", periods=5, name="Date")
    close = pd.Series([100.0, 110.0, 99.0, 99.0, 108.9], index=idx)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                         "AdjClose": close, "Volume": 1e6}, index=idx)


# ---------------------------------------------------------------- engine ---
def test_always_long_zero_cost_equals_asset(prices):
    pos = pd.Series(1.0, index=prices.index)
    res = run_backtest(prices, pos, cost_bps=0.0)
    asset = prices["AdjClose"].pct_change().fillna(0.0)
    np.testing.assert_allclose(res.returns.to_numpy(), asset.to_numpy(), rtol=1e-12)
    assert res.equity.iloc[-1] == pytest.approx(prices.AdjClose.iloc[-1] / prices.AdjClose.iloc[0])


def test_position_is_lagged_one_day():
    p = _toy()  # returns: _, +10%, -10%, 0%, +10%
    pos = pd.Series([0, 1, 0, 0, 0], index=p.index, dtype=float)  # long only at close of day 1
    r = run_backtest(p, pos, cost_bps=0.0).returns
    np.testing.assert_allclose(r.to_numpy(), [0, 0, -0.10, 0, 0], atol=1e-12)


def test_short_and_costs():
    p = _toy()
    pos = pd.Series([0, -1, -1, 0, 0], index=p.index, dtype=float)
    r = run_backtest(p, pos, cost_bps=10.0).returns
    # day2: short earns +10%, minus cost of entering at close day1 (1 * 10bp)
    # day3: short, 0% ret, no new trade at close day2
    # day4: flat, minus cost of exiting at close day3
    np.testing.assert_allclose(r.to_numpy(), [0, 0, 0.10 - 0.001, 0.0, -0.001], atol=1e-12)


def test_trade_stats():
    p = _toy()
    pos = pd.Series([0, 1, 1, 0, 0], index=p.index, dtype=float)
    s = trade_stats(run_backtest(p, pos))
    assert s["n_position_changes"] == 2
    assert s["exposure"] == pytest.approx(0.4)


def test_rejects_bad_positions():
    p = _toy()
    with pytest.raises(AssertionError):
        run_backtest(p, pd.Series([0, 2, 0, 0, 0], index=p.index, dtype=float))
    with pytest.raises(AssertionError):
        run_backtest(p, pd.Series([0, np.nan, 0, 0, 0], index=p.index))


# --------------------------------------------------------------- metrics ---
def test_metrics_hand_calculated():
    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    m = compute_metrics(r, rf_annual=0.0)
    assert set(METRIC_KEYS) <= set(m)
    assert m["mean_daily_return"] == pytest.approx(0.005)
    assert m["std_daily_return"] == pytest.approx(np.std([0.01, -0.02, 0.03, 0.0], ddof=1))
    growth = 1.01 * 0.98 * 1.03
    assert m["cumulative_return"] == pytest.approx(growth - 1)
    assert m["annualized_return"] == pytest.approx(growth ** (252 / 4) - 1)
    assert m["annualized_mean_return"] == pytest.approx(0.005 * 252)
    assert m["max_drawdown"] == pytest.approx(-0.02)
    assert m["hit_rate"] == pytest.approx(2 / 3)
    assert m["sharpe_ratio"] == pytest.approx(0.005 / m["std_daily_return"] * np.sqrt(252))


def test_max_drawdown_counts_initial_capital():
    assert max_drawdown(pd.Series([0.9, 0.95, 1.2, 0.6])) == pytest.approx(-0.5)
    assert max_drawdown(pd.Series([0.8, 0.9])) == pytest.approx(-0.2)


def test_qqq_buy_and_hold_sanity(prices):
    m = compute_metrics(prices["AdjClose"].pct_change().dropna())
    assert 0.10 < m["annualized_return"] < 0.25
    assert 0.15 < m["annualized_volatility"] < 0.35
    assert -0.60 < m["max_drawdown"] < -0.40  # 2008 crash


# -------------------------------------------------------------- features ---
@pytest.mark.parametrize("fn", [
    lambda p: F.lagged_ohlc_features(p, lags=5),
    F.technical_indicators,
])
def test_features_are_causal(fn, prices):
    full = fn(prices)
    for frac in (0.4, 0.8):
        cut = prices.index[int(len(prices) * frac)]
        trunc = fn(prices.loc[:cut])
        pd.testing.assert_frame_equal(full.loc[:cut], trunc, rtol=1e-9)


def test_rsi_bounds(prices):
    r = F.rsi(prices["AdjClose"], 14).dropna()
    assert r.between(0, 100).all()


def test_forward_return_is_future():
    p = _toy()
    fr = F.forward_return(p, 1)
    assert fr.iloc[0] == pytest.approx(0.10) and np.isnan(fr.iloc[-1])
