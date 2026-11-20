"""Tests for strategy slot 2 - NeuralNetStrategy.

(The generic contract + look-ahead tests in test_contracts.py also run on it.)
Small n_seeds / hidden sizes keep the suite fast; the defaults are tested separately.
"""
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qqq import config
from qqq.data import load_prices, split
from qqq.evaluate import check_no_lookahead
from qqq.strategies.neural_net import (
    NeuralNetStrategy,
    _MLP,
    bce_loss,
    momentum_features,
    sharpe_loss,
    train_mlp,
)

FAST = partial(NeuralNetStrategy, n_seeds=2, hidden=(8,))


@pytest.fixture(scope="module")
def qqq():
    return load_prices(source="csv")


@pytest.fixture(scope="module")
def train(qqq):
    return split(qqq)[0]


# ------------------------------------------------------------------ network ---
@pytest.mark.parametrize("loss", ["bce", "sharpe"])
def test_gradients_match_finite_differences(loss):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    y = (rng.random(120) > 0.5).astype(float)
    x = rng.normal(0, 0.01, 120)
    lf = (lambda a: bce_loss(a, y)) if loss == "bce" else (lambda a: sharpe_loss(a, x, 0.002))
    net = _MLP(5, (4, 3), seed=1, l2=0.0)
    a, acts = net.forward(X)
    gW, gb = net.grads(acts, lf(a)[1])
    for P, G in zip(net.params(), gW + gb):
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old, h = P[idx], 1e-6
            P[idx] = old + h
            lp = lf(net.forward(X)[0])[0]
            P[idx] = old - h
            lm = lf(net.forward(X)[0])[0]
            P[idx] = old
            num = (lp - lm) / (2 * h)
            assert abs(num - G[idx]) <= 1e-5 * max(1.0, abs(num)), (num, G[idx])


def test_sharpe_loss_is_scale_invariant_without_costs():
    """The reason for scale='vol': Sharpe can't pin down the exposure LEVEL."""
    rng = np.random.default_rng(1)
    a, x = rng.uniform(0.1, 0.9, 300), rng.normal(0.0005, 0.01, 300)
    assert sharpe_loss(a, x, 0.0)[0] == pytest.approx(sharpe_loss(0.3 * a, x, 0.0)[0])


def test_early_stopping_uses_the_last_rows_in_time_order():
    """The early-stopping holdout is the last 15% of rows: flipping the targets of the
    last rows changes the selected network."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 4))
    y = (rng.random(400) > 0.5).astype(float)
    base = train_mlp(X, y, "direction", (4,), seed=0, max_epochs=80, patience=10)
    y2 = y.copy()
    y2[-60:] = 1 - y2[-60:]
    flipped = train_mlp(X, y2, "direction", (4,), seed=0, max_epochs=80, patience=10)
    assert not np.allclose(base.forward(X)[0], flipped.forward(X)[0])


# ----------------------------------------------------------------- features ---
def test_momentum_features_are_causal(qqq):
    full = momentum_features(qqq)
    for cut in ("2010-06-30", "2018-12-31"):
        part = momentum_features(qqq.loc[:cut])
        pd.testing.assert_frame_equal(full.loc[:cut], part)


def test_momentum_features_warmup_then_complete(qqq):
    f = momentum_features(qqq)
    first = f.notna().all(axis=1).idxmax()
    assert f.loc[first:].notna().all().all()
    assert first < pd.Timestamp("2007-01-01")


# ----------------------------------------------------------------- strategy ---
@pytest.mark.parametrize("mode", ["direction", "sharpe"])
def test_positions_contract_and_range(mode, qqq, train):
    s = FAST(mode=mode, lags=5).fit(train)
    pos = s.generate_positions(qqq)
    assert pos.index.equals(qqq.index) and not pos.isna().any()
    assert pos.between(0, 1).all()
    if mode == "direction":
        assert set(np.unique(pos)) <= {0.0, 1.0}


def test_deterministic_given_seed(qqq, train):
    a = FAST(seed=7).fit(train).generate_positions(qqq)
    b = FAST(seed=7).fit(train).generate_positions(qqq)
    c = FAST(seed=8).fit(train).generate_positions(qqq)
    pd.testing.assert_series_equal(a, b)
    assert not a.equals(c)


def test_never_reads_data_behind_the_engines_back(qqq, train, monkeypatch):
    """fit()/generate_positions() may only use the frames they are given."""
    import qqq.data as qdata

    def boom(*a, **k):
        raise AssertionError("strategy read data on its own")

    monkeypatch.setattr(qdata, "load_prices", boom)
    monkeypatch.setattr(pd, "read_csv", boom)
    FAST().fit(train).generate_positions(qqq)


def test_target_drops_last_train_row_instead_of_labelling_it(train):
    """forward_return is NaN on the last train row; (NaN > 0) would be False (=down).
    The model must be fitted on one row fewer than the feature rows."""
    s = FAST(mode="direction", lags=5)
    X = s._features(train)
    n_feature_rows = int(X.notna().all(axis=1).sum())
    s.fit(train)
    # base-rate threshold = share of up days among rows WITH a target
    from qqq.features import forward_return
    fwd = forward_return(train, 1)
    ok = X.notna().all(axis=1) & fwd.notna()
    assert ok.sum() == n_feature_rows - 1
    s_base = FAST(mode="direction", lags=5, threshold="base").fit(train)
    assert s_base.threshold_ == pytest.approx(float((fwd[ok] > 0).mean()))


def test_scale_vol_sets_level_but_keeps_shape(qqq, train):
    raw = FAST(scale="none", band=0.0).fit(train)
    scaled = FAST(scale="vol", band=0.0).fit(train)
    assert scaled.scale_ > 1.0                     # untrained level is far below full exposure
    y = raw.raw_output(qqq).dropna()
    p = scaled.generate_positions(qqq).loc[y.index]
    np.testing.assert_allclose(p, np.minimum(1.0, scaled.scale_ * y), rtol=1e-12)


def test_band_reduces_trading(qqq, train):
    loose = FAST(band=0.0).fit(train).generate_positions(qqq)
    tight = FAST(band=0.10).fit(train).generate_positions(qqq)
    assert (tight.diff().abs() > 0).sum() < (loose.diff().abs() > 0).sum()


@pytest.mark.parametrize("kw", [{"mode": "direction", "lags": 5}, {"mode": "sharpe"},
                                {"mode": "sharpe", "scale": "vol", "band": 0.1}])
def test_variants_have_no_lookahead(kw, qqq):
    check_no_lookahead(FAST(**kw), qqq)


@pytest.mark.parametrize("kwargs", [{"mode": "classify"}, {"threshold": 1.5},
                                    {"n_seeds": 0}, {"scale": "max"}])
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        NeuralNetStrategy(**kwargs)


def test_generate_before_fit_raises(qqq):
    with pytest.raises(RuntimeError):
        NeuralNetStrategy().generate_positions(qqq)


def test_defaults_match_tuning():
    """The class defaults are the configuration the walk-forward protocol selected."""
    d = NeuralNetStrategy().params()
    assert d["mode"] == "sharpe" and d["hidden"] == [32, 16] and d["scale"] == "vol"
    assert d["turnover_bps"] == 25.0 and d["l2"] == 0.1 and d["band"] == 0.10
    assert d["n_seeds"] == 5 and d["seed"] == config.RANDOM_SEED
    res = Path(config.RESULTS) / "strategies" / "neural_net" / "v2b_scaled" / "params.json"
    if res.exists():
        assert json.loads(res.read_text()) == d
