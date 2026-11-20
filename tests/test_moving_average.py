"""Tests for strategy slot 1 - MovingAverageCrossover.

(The generic contract + look-ahead tests in test_contracts.py also run on it.)
"""
import numpy as np
import pandas as pd
import pytest

from qqq.backtest import run_backtest, trade_stats
from qqq.data import load_prices
from qqq.evaluate import check_no_lookahead
from functools import partial

from qqq.strategies.moving_average import MovingAverageCrossover

# Plain crossover (no re-entry, no vol cap) - the tuned defaults turn both on.
MA = partial(MovingAverageCrossover, reentry_window=None, vol_target=None)


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


def reference_positions(prices, short, long_, ma_type="sma", band=0.0, allow_short=False,
                        reentry=None):
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
    rr = ma(reentry) if reentry else None
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
        if rr is not None and not np.isnan(rr[t]) and p[t] > rr[t]:
            pos[t] = 1.0
    return pd.Series(pos, index=prices.index)


# ------------------------------------------------------------------ toy data ---
def test_crossover_day_and_warmup():
    pos = MA(2, 4).generate_positions(_prices(V))
    assert pos.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_allow_short():
    pos = MA(2, 4, allow_short=True).generate_positions(_prices(V))
    assert pos.tolist() == [0, 0, 0, -1, -1, -1, 1, 1, 1, 1]


def test_engine_trades_the_day_after_the_signal():
    p = _prices(V)
    res = run_backtest(p, MA(2, 4).generate_positions(p), cost_bps=0.0, cash_rate=None)
    # signal at close of day 6 -> first P&L on day 7 (+9/8 - 1)
    assert res.returns.iloc[:7].abs().sum() == 0
    assert res.returns.iloc[7] == pytest.approx(9 / 8 - 1)


def test_hysteresis_keeps_position_inside_band():
    # price vs SMA3 ratio: day3 +3.9% (enter), day4 +1.3%, day5 -0.95% (both inside
    # a 2% band -> hold), day6 -6.3% (exit).
    p = _prices([100, 100, 100, 106, 105, 104, 95])
    ratio = MA(1, 3).moving_averages(p)["ratio"]
    assert -0.02 < ratio.iloc[5] < 0 < ratio.iloc[4] < 0.02
    pos_band = MA(1, 3, band=0.02).generate_positions(p)
    pos_none = MA(1, 3, band=0.0).generate_positions(p)
    assert pos_band.iloc[3:].tolist() == [1, 1, 1, 0]  # band holds through day 5
    assert pos_none.iloc[3:].tolist() == [1, 1, 0, 0]  # no band: exits at the crossover


def test_short_window_one_is_price_vs_ma():
    p = _prices(V)
    ma = MA(1, 4).moving_averages(p)
    pd.testing.assert_series_equal(ma["ma_short"], p["AdjClose"], check_names=False)


@pytest.mark.parametrize("kwargs", [
    dict(short_window=50, long_window=50),
    dict(short_window=0, long_window=10),
    dict(short_window=5, long_window=10, ma_type="wma"),
    dict(short_window=5, long_window=10, band=-0.01),
    dict(short_window=5.5, long_window=10),
    dict(short_window=5, long_window=10, reentry_window=10),
    dict(short_window=5, long_window=10, reentry_window=1),
])
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        MA(**kwargs)


def test_reentry_goes_long_before_the_crossover():
    # price climbs back above SMA2 on day 5, one day before SMA2 crosses SMA4
    pos = MA(2, 4, reentry_window=2).generate_positions(_prices(V))
    assert pos.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_reentry_overrides_short():
    pos = MA(2, 4, allow_short=True, reentry_window=2).generate_positions(_prices(V))
    assert pos.tolist() == [0, 0, 0, -1, -1, 1, 1, 1, 1, 1]


def test_reentry_respects_long_window_warmup():
    # price > SMA2 from day 1, but SMA4 only exists from day 3
    pos = MA(2, 4, reentry_window=2).generate_positions(_prices(range(1, 9)))
    assert pos.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]


def test_reentry_never_reduces_exposure(qqq):
    base = MA(50, 200).generate_positions(qqq)
    re = MA(50, 200, reentry_window=50).generate_positions(qqq)
    assert (re >= base).all() and (re > base).any()


# ------------------------------------------------------------------ real QQQ ---
@pytest.mark.parametrize("kw", [
    dict(short=50, long_=200),
    dict(short=1, long_=200),
    dict(short=10, long_=50, ma_type="ema"),
    dict(short=20, long_=100, band=0.01),
    dict(short=10, long_=50, band=0.005, allow_short=True),
    dict(short=50, long_=200, reentry=50),
    dict(short=20, long_=150, ma_type="ema", band=0.01, reentry=20),
    dict(short=10, long_=100, allow_short=True, reentry=30),
])
def test_matches_reference_on_qqq(kw, qqq):
    strat = MovingAverageCrossover(kw["short"], kw["long_"], kw.get("ma_type", "sma"),
                                   kw.get("band", 0.0), kw.get("allow_short", False),
                                   reentry_window=kw.get("reentry"), vol_target=None)
    got = strat.generate_positions(qqq)
    exp = reference_positions(qqq, **kw)
    pd.testing.assert_series_equal(got, exp, check_names=False)


def test_band_reduces_trading(qqq):
    n0 = trade_stats(run_backtest(qqq, MA(10, 50).generate_positions(qqq)))
    n2 = trade_stats(run_backtest(qqq, MA(10, 50, band=0.02)
                                  .generate_positions(qqq)))
    assert n2["n_position_changes"] < n0["n_position_changes"]


@pytest.mark.parametrize("strat", [
    MA(20, 100, ma_type="ema", band=0.01),
    MA(50, 200, ma_type="ema", band=0.01, reentry_window=50),
    MA(5, 50, allow_short=True),
])
def test_variants_have_no_lookahead(strat, qqq):
    check_no_lookahead(strat, qqq)


def test_defaults_match_tuning():
    """Defaults = choice in notebooks/01_moving_average/tuning.ipynb. Update both together."""
    assert MovingAverageCrossover().params() == {
        "short_window": 50, "long_window": 250, "ma_type": "sma", "band": 0.0,
        "allow_short": False, "reentry_window": 50,
        "vol_target": "train", "vol_window": 20, "vol_band": 0.1,
    }


# ------------------------------------------------------------- vol cap ---
def reference_vol_cap(signal, prices, target, window, band):
    """Independent re-implementation of the volatility cap."""
    r = prices["AdjClose"].pct_change()
    vol = r.rolling(window, min_periods=window).std() * np.sqrt(252)
    out, cur, prev = [], 0.0, 0.0
    for t in range(len(signal)):
        s = signal.iloc[t]
        scale = 1.0 if np.isnan(vol.iloc[t]) else min(1.0, target / vol.iloc[t])
        x = s * scale
        if s != prev or x == 0 or abs(x) == 1 or abs(x - cur) > band:
            cur = x
        out.append(cur)
        prev = s
    return pd.Series(out, index=signal.index)


@pytest.mark.parametrize("target,band", [(0.20, 0.0), (0.25, 0.2), (0.15, 0.1)])
def test_vol_cap_matches_reference(target, band, qqq):
    base = MovingAverageCrossover(vol_target=None).generate_positions(qqq)
    got = MovingAverageCrossover(vol_target=target, vol_band=band).generate_positions(qqq)
    exp = reference_vol_cap(base, qqq, target, 20, band)
    pd.testing.assert_series_equal(got, exp, check_names=False)


def test_vol_cap_never_levers_or_flips(qqq):
    base = MovingAverageCrossover(vol_target=None, allow_short=True,
                                  reentry_window=None).generate_positions(qqq)
    capped = MovingAverageCrossover(vol_target=0.20, vol_band=0.1, allow_short=True,
                                    reentry_window=None).generate_positions(qqq)
    assert (capped.abs() <= base.abs() + 1e-12).all()
    assert (np.sign(capped) == np.sign(base)).all()


def test_vol_cap_cuts_exposure_in_2008(qqq):
    capped = MovingAverageCrossover(vol_target=0.20, vol_band=0.0).generate_positions(qqq)
    base = MovingAverageCrossover(vol_target=None).generate_positions(qqq)
    calm, storm = slice("2013-01-01", "2013-12-31"), slice("2008-09-15", "2008-12-31")
    assert capped[calm].mean() == pytest.approx(base[calm].mean(), abs=0.05)  # calm: ~unchanged
    assert capped[storm].sum() < 0.6 * base[storm].sum() or base[storm].sum() == 0


def test_vol_band_reduces_trading(qqq):
    n = {b: trade_stats(run_backtest(qqq, MovingAverageCrossover(vol_target=0.25, vol_band=b)
                                     .generate_positions(qqq)))["n_position_changes"]
         for b in (0.0, 0.1, 0.2)}
    assert n[0.0] > n[0.1] > n[0.2]


def test_vol_target_train_is_fitted_on_train_only(qqq):
    from qqq.data import split
    train, _, _ = split(qqq)
    s = MovingAverageCrossover(vol_target="train")
    with pytest.raises(ValueError):
        s.generate_positions(qqq)
    s.fit(train)
    expected = train["AdjClose"].pct_change().std() * np.sqrt(252)
    assert s.fitted_vol_target_ == pytest.approx(expected)
    assert 0.18 < s.fitted_vol_target_ < 0.23
    s2 = MovingAverageCrossover(vol_target="train").fit(qqq)  # different data -> different target
    assert s2.fitted_vol_target_ != pytest.approx(expected)


@pytest.mark.parametrize("kwargs", [dict(vol_target=0), dict(vol_target="test"),
                                    dict(vol_target=0.2, vol_band=1.0),
                                    dict(vol_target=0.2, vol_window=1)])
def test_invalid_vol_params(kwargs):
    with pytest.raises(ValueError):
        MovingAverageCrossover(**kwargs)


@pytest.mark.parametrize("strat", [MovingAverageCrossover(vol_target=0.25, vol_band=0.2),
                                   MovingAverageCrossover(vol_target="train", vol_band=0.1)])
def test_vol_cap_has_no_lookahead(strat, qqq):
    check_no_lookahead(strat, qqq)
