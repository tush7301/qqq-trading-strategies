"""Strategy slot 2 - Neural network. Owner: akshatbhandari15

A small feed-forward neural network, written in NumPy (no deep-learning dependency,
so CI and every teammate can run it). One network class, two ways to train it:

mode="direction"  (v1, the brief's design)
    inputs : lagged OHLC features (returns, range, body, gap, volume z), `lags` days
    target : 1 if tomorrow's return > 0 else 0          (built inside fit(), train only)
    loss   : binary cross-entropy
    trade  : long if the seed-averaged P(up) > threshold, else cash

mode="sharpe"     (Deep-Momentum-Network style; Lim, Zohren & Roberts 2019)
    inputs : volatility-normalised returns over 1d/1m/3m/6m/12m, three MACD trend
             signals (Baz et al. 2015), and log volatility
    output : the position itself, in [0, 1] (sigmoid)
    loss   : -Sharpe ratio of  position_t * r_{t+1} - c * |position_t - position_{t-1}|
             on the training window; c is a turnover penalty in bps
    trade  : the seed-averaged output, optionally rebalanced only when it moves by
             more than `band`

scale="vol"       (v2b) Sharpe is unchanged by scaling every position by a constant,
                  so the output LEVEL of a Sharpe-trained net is arbitrary. fit() sets
                  k = QQQ train vol / train vol of the net's own strategy, and the
                  position is min(1, k * output) (Lim et al. rescale to a vol target).

Defaults = v2b, chosen by the pre-registered walk-forward protocol on TRAIN only
(notebooks/02_neural_net/PROTOCOL.md; results in results/strategies/neural_net/):
    mode="sharpe", hidden=(32, 16), turnover_bps=25, l2=0.1, band=0.10, scale="vol"
v1 (the brief's classifier) = mode="direction", lags=20, hidden=(32, 16), threshold=0.5.

Leakage controls (see notebooks/02_neural_net/PROTOCOL.md)
  * Every feature at t uses data up to the close of t (tested by truncation).
  * Targets / next-day returns are computed inside fit() from the TRAIN frame only;
    the last train row has no target and is dropped (never labelled 0).
  * Standardisation statistics come from the train frame only.
  * Early stopping holds out the LAST 15% of the train window, in time order.
  * Positions are not shifted here; the engine trades position[t] from t to t+1.
  * Every configuration is trained with `n_seeds` seeds and the outputs averaged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qqq import config
from qqq import features as F
from qqq.strategies.base import Strategy

MODES = ("direction", "sharpe")
_SQRT252 = np.sqrt(config.TRADING_DAYS_PER_YEAR)


# ------------------------------------------------------------------ features ---
def momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Deep-Momentum-Network inputs (Lim et al. 2019), all causal and scale-free.

    nret_h : return over the last h days / (daily vol * sqrt(h)),  h in 1,21,63,126,252
    macd_S_L : (EMA_S - EMA_L) / 63d price std, then / its own 252d std  (Baz et al.)
    log_vol : log of annualised 60-day EWM volatility
    """
    p = prices["AdjClose"]
    r = p.pct_change()
    sigma = r.ewm(span=60, adjust=False, min_periods=60).std()
    out = {}
    for h in (1, 21, 63, 126, 252):
        out[f"nret_{h}"] = (p / p.shift(h) - 1) / (sigma * np.sqrt(h))
    for s, lng in ((8, 24), (16, 48), (32, 96)):
        q = (F.ema(p, s) - F.ema(p, lng)) / p.rolling(63, min_periods=63).std()
        out[f"macd_{s}_{lng}"] = q / q.rolling(252, min_periods=252).std()
    out["log_vol"] = np.log(sigma * _SQRT252)
    return pd.DataFrame(out, index=prices.index)


def direction_features(prices: pd.DataFrame, lags: int) -> pd.DataFrame:
    return F.lagged_ohlc_features(prices, lags=lags)


# ------------------------------------------------------------------- network ---
class _MLP:
    """tanh hidden layers -> sigmoid output. Full-batch Adam, L2 weight decay."""

    def __init__(self, n_in: int, hidden: tuple[int, ...], seed: int, l2: float):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.W = [rng.normal(0.0, np.sqrt(1.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.l2 = l2

    def forward(self, X: np.ndarray):
        acts = [X]
        h = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.tanh(h @ W + b)
            acts.append(h)
        z = (h @ self.W[-1] + self.b[-1]).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), acts

    def grads(self, acts, dz: np.ndarray):
        delta = dz[:, None]
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta + self.l2 * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (1.0 - acts[i] ** 2)
        return gW, gb

    def params(self):
        return self.W + self.b

    def set_params(self, ps):
        k = len(self.W)
        self.W, self.b = [p.copy() for p in ps[:k]], [p.copy() for p in ps[k:]]


def bce_loss(a: np.ndarray, y: np.ndarray):
    """Binary cross-entropy and dL/dz for a sigmoid output a."""
    eps = 1e-12
    loss = -np.mean(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps))
    return loss, (a - y) / len(y)


def sharpe_loss(a: np.ndarray, x: np.ndarray, cost: float):
    """-annualised Sharpe of R_t = a_t * x_t - cost * |a_t - a_{t-1}|, and dL/dz.

    a: positions (sigmoid outputs) in time order; x: next-day returns aligned with a.
    """
    m = len(a)
    da = np.diff(a, prepend=a[0])
    R = a * x - cost * np.abs(da)
    mu = R.mean()
    s = np.sqrt(max((R ** 2).mean() - mu ** 2, 1e-12))
    loss = -_SQRT252 * mu / s
    dR = -_SQRT252 * (1.0 / (m * s) - mu * (R - mu) / (m * s ** 3))
    sg = np.sign(da)
    dA = dR * (x - cost * sg)
    dA[:-1] += dR[1:] * cost * sg[1:]          # R_{t+1} depends on a_t through |a_{t+1}-a_t|
    return loss, dA * a * (1.0 - a)


def train_mlp(X, target, mode, hidden, seed, l2=1e-3, lr=3e-3, max_epochs=600,
              patience=60, holdout=0.15, cost=0.0) -> _MLP:
    """Full-batch Adam with early stopping on the LAST `holdout` share of rows (time order)."""
    n_hold = max(int(len(X) * holdout), 50)
    Xf, Xh, tf, th = X[:-n_hold], X[-n_hold:], target[:-n_hold], target[-n_hold:]

    def loss_fn(a, t):
        return bce_loss(a, t) if mode == "direction" else sharpe_loss(a, t, cost)

    net = _MLP(X.shape[1], hidden, seed, l2)
    m = [np.zeros_like(p) for p in net.params()]
    v = [np.zeros_like(p) for p in net.params()]
    b1, b2 = 0.9, 0.999
    best, best_ps, wait = np.inf, [p.copy() for p in net.params()], 0
    for epoch in range(1, max_epochs + 1):
        a, acts = net.forward(Xf)
        _, dz = loss_fn(a, tf)
        gW, gb = net.grads(acts, dz)
        ps = net.params()
        for i, (p, g) in enumerate(zip(ps, gW + gb)):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g ** 2
            p -= lr * (m[i] / (1 - b1 ** epoch)) / (np.sqrt(v[i] / (1 - b2 ** epoch)) + 1e-8)
        hold_loss = loss_fn(net.forward(Xh)[0], th)[0]
        if hold_loss < best - 1e-6:
            best, best_ps, wait = hold_loss, [p.copy() for p in net.params()], 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.set_params(best_ps)
    return net


# ------------------------------------------------------------------ strategy ---
class NeuralNetStrategy(Strategy):
    name = "neural_net"

    def __init__(
        self,
        mode: str = "sharpe",
        lags: int = 20,
        hidden: tuple[int, ...] = (32, 16),
        threshold: float | str = 0.5,
        turnover_bps: float = 25.0,
        band: float = 0.10,
        scale: str = "vol",
        l2: float = 0.1,
        n_seeds: int = 5,
        seed: int = config.RANDOM_SEED,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not (isinstance(threshold, (int, float)) and 0 < threshold < 1) and threshold != "base":
            raise ValueError("threshold must be in (0, 1) or 'base' (train share of up days)")
        if n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")
        if scale not in ("none", "vol"):
            raise ValueError("scale must be 'none' or 'vol'")
        self.mode = mode
        self.lags = int(lags)
        self.hidden = tuple(int(h) for h in hidden)
        self.threshold = threshold
        self.turnover_bps = float(turnover_bps)
        self.band = float(band)
        self.scale = scale
        self.l2 = float(l2)
        self.n_seeds = int(n_seeds)
        self.seed = int(seed)
        self.models: list[_MLP] = []
        self.mu = self.sd = None
        self.threshold_ = None
        self.scale_ = 1.0

    # ---------------------------------------------------------------- helpers ---
    def _features(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "direction":
            return direction_features(prices, self.lags)
        return momentum_features(prices)

    def _standardise(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(((X - self.mu) / self.sd).to_numpy(), -5.0, 5.0)

    # -------------------------------------------------------------------- fit ---
    def fit(self, train_prices: pd.DataFrame) -> "NeuralNetStrategy":
        X = self._features(train_prices)
        nxt = F.forward_return(train_prices, 1)            # target: built from TRAIN only
        rows = X.notna().all(axis=1) & nxt.notna()          # warm-up and last row dropped
        idx = np.flatnonzero(rows.to_numpy())
        if len(idx) < 250:
            raise ValueError("not enough training rows after warm-up")
        if self.mode == "sharpe" and not (np.diff(idx) == 1).all():
            raise ValueError("sharpe mode needs contiguous training days (turnover term)")
        Xtr = X[rows]
        self.mu, self.sd = Xtr.mean(), Xtr.std(ddof=0).replace(0.0, 1.0)
        Z = self._standardise(Xtr)
        x = nxt[rows].to_numpy()
        target = (x > 0).astype(float) if self.mode == "direction" else x
        self.threshold_ = float((x > 0).mean()) if self.threshold == "base" else self.threshold
        cost = self.turnover_bps / 1e4
        self.models = [
            train_mlp(Z, target, self.mode, self.hidden, self.seed + k, l2=self.l2, cost=cost)
            for k in range(self.n_seeds)
        ]
        self.scale_ = 1.0
        if self.mode == "sharpe" and self.scale == "vol":
            # Sharpe is unchanged by scaling every position by a constant, so the
            # output level is arbitrary. Rescale so the train-period strategy vol
            # equals QQQ's train-period vol (Lim et al. 2019 rescale to a vol target).
            a = np.mean([m.forward(Z)[0] for m in self.models], axis=0)
            self.scale_ = float(np.std(x) / max(np.std(a * x), 1e-12))
        return self

    # ---------------------------------------------------------------- predict ---
    def raw_output(self, prices: pd.DataFrame) -> pd.Series:
        """Seed-averaged network output (P(up) or position), NaN during warm-up."""
        if not self.models:
            raise RuntimeError("call fit() first")
        X = self._features(prices)
        ok = X.notna().all(axis=1)
        out = pd.Series(np.nan, index=prices.index)
        if ok.any():
            Z = self._standardise(X[ok])
            out[ok] = np.mean([m.forward(Z)[0] for m in self.models], axis=0)
        return out

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        y = self.raw_output(prices)
        if self.mode == "direction":
            pos = (y > self.threshold_).astype(float)
        else:
            pos = (self.scale_ * y).clip(upper=1.0).fillna(0.0)
        pos = pos.where(y.notna(), 0.0)
        if self.band > 0:
            vals, cur = [], 0.0
            for v in pos.to_numpy():                        # rebalance only on big moves
                if abs(v - cur) > self.band or (v == 0.0) != (cur == 0.0):
                    cur = v
                vals.append(cur)
            pos = pd.Series(vals, index=prices.index)
        return pos.astype(float).clip(0.0, 1.0).rename("position")

    def params(self) -> dict:
        return {"mode": self.mode, "lags": self.lags, "hidden": list(self.hidden),
                "threshold": self.threshold, "turnover_bps": self.turnover_bps,
                "band": self.band, "scale": self.scale, "l2": self.l2,
                "n_seeds": self.n_seeds, "seed": self.seed}
