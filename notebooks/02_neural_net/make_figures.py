"""Figures and tables for the Slot 2 write-up (train + validation only).

    python notebooks/02_neural_net/make_figures.py

Needs results from nn_research.py. Writes notebooks/02_neural_net/figures/*.png and
results/strategies/neural_net/research/stress_episodes.csv.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qqq import config  # noqa: E402
from qqq import features as F  # noqa: E402
from qqq.backtest import run_backtest  # noqa: E402
from qqq.data import load_prices  # noqa: E402
from qqq.strategies.base import BuyAndHold  # noqa: E402
from qqq.strategies.moving_average import MovingAverageCrossover  # noqa: E402
from qqq.strategies.neural_net import (  # noqa: E402
    NeuralNetStrategy,
    _MLP,
    bce_loss,
    sharpe_loss,
)

FIG = Path(__file__).resolve().parent / "figures"
RES = config.RESULTS / "strategies" / "neural_net" / "research"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"            # categorical slots 1-3
V1 = {"mode": "direction", "lags": 20, "hidden": (32, 16), "threshold": 0.5, "l2": 0.01,
      "band": 0.0, "scale": "none"}
V2 = {"scale": "none"}
V2B: dict = {}


def _style(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def overfitting_curves(prices):
    """Train vs time-ordered holdout loss without early stopping, fit window 2005-2016."""
    fit = prices.loc[:"2016-12-31"]
    curves = {}
    for mode in ("direction", "sharpe"):
        s = NeuralNetStrategy(**(V1 if mode == "direction" else V2))
        X = s._features(fit)
        nxt = F.forward_return(fit, 1)
        ok = X.notna().all(axis=1) & nxt.notna()
        s.mu, s.sd = X[ok].mean(), X[ok].std(ddof=0).replace(0, 1)
        Z, x = s._standardise(X[ok]), nxt[ok].to_numpy()
        t = (x > 0).astype(float) if mode == "direction" else x
        n_h = int(len(Z) * 0.15)
        net = _MLP(Z.shape[1], (32, 16), config.RANDOM_SEED, 0.01 if mode == "direction" else 0.1)

        def lf(a, tt, m=mode):
            return bce_loss(a, tt) if m == "direction" else sharpe_loss(a, tt, 25e-4)

        m_, v_ = [np.zeros_like(p) for p in net.params()], [np.zeros_like(p) for p in net.params()]
        tr_l, ho_l = [], []
        for ep in range(1, 601):
            a, acts = net.forward(Z[:-n_h])
            loss, dz = lf(a, t[:-n_h])
            gW, gb = net.grads(acts, dz)
            for i, (p, g) in enumerate(zip(net.params(), gW + gb)):
                m_[i] = 0.9 * m_[i] + 0.1 * g
                v_[i] = 0.999 * v_[i] + 0.001 * g ** 2
                p -= 3e-3 * (m_[i] / (1 - 0.9 ** ep)) / (np.sqrt(v_[i] / (1 - 0.999 ** ep)) + 1e-8)
            tr_l.append(loss)
            ho_l.append(lf(net.forward(Z[-n_h:])[0], t[-n_h:])[0])
        curves[mode] = (np.array(tr_l), np.array(ho_l))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    tr, ho = curves["direction"]
    ep = np.arange(1, len(tr) + 1)
    axes[0].plot(ep, tr, color=S1, lw=2, label="Training rows (2005–2014)")
    axes[0].plot(ep, ho, color=S2, lw=2, label="Held-out last 15% (2015–2016)")
    axes[0].axhline(np.log(2), color=INK2, lw=1, ls="--")
    axes[0].text(590, np.log(2) - 0.05, "coin flip (ln 2)", color=INK2, fontsize=8, ha="right")
    _style(axes[0], "Direction classifier: cross-entropy", "loss (lower is better)")
    axes[0].set_xlabel("epoch", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    tr, ho = curves["sharpe"]
    axes[1].plot(ep, -tr, color=S1, lw=2, label="Training rows (in-sample)")
    axes[1].plot(ep, -ho, color=S2, lw=2, label="Held-out last 15%")
    _style(axes[1], "Sharpe-trained network: Sharpe ratio", "annualised Sharpe")
    axes[1].set_xlabel("epoch", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "overfitting_curves.png", dpi=150)
    plt.close(fig)
    return {k: {"best_holdout_epoch": int(np.argmin(v[1]) + 1),
                "train_at_600": float(v[0][-1]), "holdout_at_600": float(v[1][-1])}
            for k, v in curves.items()}


def what_it_learned(prices):
    s = NeuralNetStrategy(**V2B).fit(prices.loc[:config.TRAIN_END])
    pos = s.generate_positions(prices)
    raw = s.raw_output(prices)
    r = prices["AdjClose"].pct_change()
    vol = r.ewm(span=60, adjust=False, min_periods=60).std() * np.sqrt(252)
    mom = prices["AdjClose"] / prices["AdjClose"].shift(63) - 1
    df = pd.DataFrame({"pos": (s.scale_ * raw).clip(upper=1), "vol": vol, "mom": mom}).dropna()
    df = df.loc[:config.TRAIN_END]

    fig = plt.figure(figsize=(11, 7))
    ax0 = fig.add_subplot(2, 1, 1)
    ax0.fill_between(pos.index, 0, pos, color=S1, alpha=0.35, lw=0, step="post",
                     label="Position (share of capital in QQQ)")
    ax0.axvspan(pd.Timestamp(config.VALIDATION_START), pd.Timestamp(config.VALIDATION_END),
                color=GRID, alpha=0.6, lw=0)
    ax0.text(pd.Timestamp("2019-02-01"), 0.03, "validation", color=INK2, fontsize=8)
    ax0.set_ylim(0, 1.1)
    for a, b, lab in (("2008-09-01", "2009-03-31", "2008"), ("2020-02-20", "2020-04-30", "COVID")):
        ax0.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=S2, alpha=0.15, lw=0)
        ax0.text(pd.Timestamp(a), 1.03, lab, color=INK2, fontsize=8)
    _style(ax0, "v2b position over time (fitted on 2005–2018; orange = crash periods)",
           "share of capital in QQQ")

    for k, (col, label, fmt) in enumerate((("vol", "60-day volatility (annualised)", "{:.0%}"),
                                           ("mom", "Trailing 3-month return", "{:+.0%}"))):
        ax = fig.add_subplot(2, 2, 3 + k)
        bins = pd.qcut(df[col], 10)
        g = df.groupby(bins, observed=True)["pos"].mean()
        mids = [b.mid for b in g.index]
        ax.plot(mids, g.values, color=S1, lw=2, marker="o", ms=5)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _, f=fmt: f.format(v)))
        ax.set_ylim(0, 1.05)
        _style(ax, f"Average position by decile of {label.lower()}", "average position")
        ax.set_xlabel(f"{label} (train period, decile midpoints)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "what_it_learned.png", dpi=150)
    plt.close(fig)
    corr = df.corr()["pos"].round(2).to_dict()
    return {"scale_k": s.scale_, "corr_with_position_train": corr}


def family_comparison(prices):
    wf = pd.read_csv(RES / "best_per_family.csv").set_index("family")["sharpe"]
    val = pd.read_csv(RES / "validation_once.csv", index_col=0)["val_sharpe"]
    v2b = pd.read_csv(RES / "v2b_scaled_posthoc.csv", index_col=0).iloc[0]
    labels = {"buy_and_hold": "Buy & hold", "A": "A  Direction MLP (v1)",
              "B": "B  Direction LSTM", "C": "C  Sharpe MLP (v2)", "C-LSTM": "C  Sharpe LSTM",
              "D": "D  NN vol forecast sizing", "D-baseline": "D  20-day vol sizing (no NN)"}
    from nn_research import walk_forward
    rows = []
    bh_wf = walk_forward(BuyAndHold, prices)["sharpe"]
    for k, lab in labels.items():
        rows.append((lab, bh_wf if k == "buy_and_hold" else wf.get(k, np.nan), val.get(k, np.nan)))
    rows.append(("C  Sharpe MLP, rescaled (v2b)*", float(v2b["wf_sharpe"]),
                 float(v2b["val_sharpe"])))
    df = pd.DataFrame(rows, columns=["label", "wf", "val"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(df))
    ax.barh(y + 0.2, df["wf"], height=0.36, color=S1, label="Walk-forward 2011–2018 (train only)")
    ax.barh(y - 0.2, df["val"], height=0.36, color=S2, label="Validation 2019–2021 (seen once)")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    for yy, (w, v) in zip(y, zip(df["wf"], df["val"])):
        ax.text(w + 0.01, yy + 0.2, f"{w:.2f}", va="center", fontsize=8, color=INK2)
        ax.text(v + 0.01, yy - 0.2, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 1.75)
    _style(ax, "Neural-network families vs buy-and-hold: Sharpe (excess of T-bills)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.07), ncol=2)
    fig.text(0.01, 0.01, "* rescaling added after the first validation look (post-hoc)",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "family_comparison.png", dpi=150)
    plt.close(fig)


def stress_episodes(prices):
    eps = {"Financial crisis (31 Oct 2007 – 9 Mar 2009)*": ("2007-10-31", "2009-03-09"),
           "Q4 2018 sell-off (1 Oct – 24 Dec 2018)*": ("2018-10-01", "2018-12-24"),
           "COVID crash (20 Feb – 23 Mar 2020)": ("2020-02-20", "2020-03-23"),
           "COVID crash and rebound (20 Feb – 31 Dec 2020)": ("2020-02-20", "2020-12-31")}
    strats = {"v1 direction": NeuralNetStrategy(**V1), "v2 Sharpe": NeuralNetStrategy(**V2),
              "v2b Sharpe, rescaled": NeuralNetStrategy(**V2B),
              "Slot 1 MA v3": MovingAverageCrossover(), "Buy & hold": BuyAndHold()}
    out = {}
    for name, s in strats.items():
        s.fit(prices.loc[:config.TRAIN_END])
        r = run_backtest(prices, s.generate_positions(prices)).returns
        out[name] = {e: float((1 + r.loc[a:b]).prod() - 1) for e, (a, b) in eps.items()}
    df = pd.DataFrame(out)
    df.to_csv(RES / "stress_episodes.csv")
    return df


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    FIG.mkdir(parents=True, exist_ok=True)
    P = load_prices(source="csv", end=config.VALIDATION_END)
    print("overfitting:", overfitting_curves(P))
    print("learned:", what_it_learned(P))
    family_comparison(P)
    print(stress_episodes(P).map(lambda v: f"{v:+.1%}").to_string())
