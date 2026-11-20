"""Tests for strategy slot 1 - MovingAverageCrossover.

(The generic contract + look-ahead tests in test_contracts.py also run on it.)
"""
import numpy as np
import pandas as pd
import pytest

from qqq.backtest import run_backtest, trade_stats
from qqq.data import load_prices
from qqq.evaluate import check_no_lookahead
from qqq.strategies.moving_average import MovingAverageCrossover


def _prices(close):
    idx = pd.bdate_range("2020-01-01", periods=len(close), name="Date")
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "AdjClose": c,
                         "Volume": 1e6}, index=idx)


# V-shape: falls for 4 days, then rises.
V = [10, 9, 8, 7, 6, 7, 8, 9, 10, 11]
# SMA2 vs SMA4 (first SMA4 on day 3):
#   day 3: 7.5 vs 8.5  (below)    day 6: 7.5 vs 7.0 (above -> first long)
#   day 4: 6.5 vs 7.5  (below)    day 7..9: above
#   day 5: 6.5 vs 7.0  (below)


@pytest.fixture(scope="module")
def qqq():
    return load_prices(source="csv")


def reference_positions(prices, short, long_, ma_type="sma", band=0.0, allow_short=False):
    """Deliberately naive day-by-day loop - an independent re-implementation."""
    p = prices["AdjClose"].to_numpy()
    n = len(p)

    def ma(w):
        out = np.full(n, np.nan)
        if w == 1:
            return p.copy()
        if ma_type == "sma":
            for t in range(w - 1, n):
                out[t] = p[t - w + 1 : t + 1].mean()
        else:
            alpha, e = 2 / (w + 1), p[0]
            for t in range(n):
                e = p[t] if t == 0 else alpha * p[t] + (1 - alpha) * e
                if t >= w - 1:
                    out[t] = e
        return out

    s, l_ = ma(short), ma(long_)
    pos, prev = np.zeros(n), 0.0
    for t in range(n):
        if np.isnan(s[t]) or np.isnan(l_[t]):
            pos[t] = 0.0
            continue
        r = s[t] / l_[t] - 1
        if r > band:
            prev = 1.0
        elif r < -band:
            prev = -1.0 if allow_short else 0.0
        pos[t] = prev
    return pd.Series(pos, index=prices.index)


# ------------------------------------------------------------------ toy data ---
def test_crossover_day_and_warmup():
    pos = MovingAverageCrossover(2, 4).generate_positions(_prices(V))
    assert pos.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_allow_short():
    pos = MovingAverageCrossover(2, 4, allow_short=True).generate_positions(_prices(V))
    assert pos.tolist() == [0, 0, 0, -1, -1, -1, 1, 1, 1, 1]


def test_engine_trades_the_day_after_the_signal():
    p = _prices(V)
    res = run_backtest(p, MovingAverageCrossover(2, 4).generate_positions(p), cost_bps=0.0)
    # signal at close of day 6 -> first P&L on day 7 (+9/8 - 1)
    assert res.returns.iloc[:7].abs().sum() == 0
    assert res.returns.iloc[7] == pytest.approx(9 / 8 - 1)


def test_hysteresis_keeps_position_inside_band():
    # price vs SMA3 ratio: day3 +3.9% (enter), day4 +1.3%, day5 -0.95% (both inside
    # a 2% band -> hold), day6 -6.3% (exit).
    p = _prices([100, 100, 100, 106, 105, 104, 95])
    ratio = MovingAverageCrossover(1, 3).moving_averages(p)["ratio"]
    assert -0.02 < ratio.iloc[5] < 0 < ratio.iloc[4] < 0.02
    pos_band = MovingAverageCrossover(1, 3, band=0.02).generate_positions(p)
    pos_none = MovingAverageCrossover(1, 3, band=0.0).generate_positions(p)
    assert pos_band.iloc[3:].tolist() == [1, 1, 1, 0]  # band holds through day 5
    assert pos_none.iloc[3:].tolist() == [1, 1, 0, 0]  # no band: exits at the crossover


def test_short_window_one_is_price_vs_ma():
    p = _prices(V)
    ma = MovingAverageCrossover(1, 4).moving_averages(p)
    pd.testing.assert_series_equal(ma["ma_short"], p["AdjClose"], check_names=False)


@pytest.mark.parametrize("kwargs", [
    dict(short_window=50, long_window=50),
    dict(short_window=0, long_window=10),
    dict(short_window=5, long_window=10, ma_type="wma"),
    dict(short_window=5, long_window=10, band=-0.01),
    dict(short_window=5.5, long_window=10),
])
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        MovingAverageCrossover(**kwargs)


# ------------------------------------------------------------------ real QQQ ---
@pytest.mark.parametrize("kw", [
    dict(short=50, long_=200),
    dict(short=1, long_=200),
    dict(short=10, long_=50, ma_type="ema"),
    dict(short=20, long_=100, band=0.01),
    dict(short=10, long_=50, band=0.005, allow_short=True),
])
def test_matches_reference_on_qqq(kw, qqq):
    strat = MovingAverageCrossover(kw["short"], kw["long_"], kw.get("ma_type", "sma"),
                                   kw.get("band", 0.0), kw.get("allow_short", False))
    got = strat.generate_positions(qqq)
    exp = reference_positions(qqq, **kw)
    pd.testing.assert_series_equal(got, exp, check_names=False)


def test_band_reduces_trading(qqq):
    n0 = trade_stats(run_backtest(qqq, MovingAverageCrossover(10, 50).generate_positions(qqq)))
    n2 = trade_stats(run_backtest(qqq, MovingAverageCrossover(10, 50, band=0.02)
                                  .generate_positions(qqq)))
    assert n2["n_position_changes"] < n0["n_position_changes"]


@pytest.mark.parametrize("strat", [
    MovingAverageCrossover(20, 100, ma_type="ema", band=0.01),
    MovingAverageCrossover(5, 50, allow_short=True),
])
def test_variants_have_no_lookahead(strat, qqq):
    check_no_lookahead(strat, qqq)
