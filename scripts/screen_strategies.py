"""Strategy screen: textbook versions of widely used systematic strategies on QQQ.

    python scripts/screen_strategies.py            # ~30 s; prints tables, writes results/research/

Companion to docs/STRATEGY_RESEARCH.md. Every strategy uses the parameters from the
literature it comes from (NO tuning), runs through the same engine as the strategy
slots (5 bps costs, cash earns T-bills, Sharpe in excess of T-bills), and is reported on
TRAIN and VALIDATION only. The test period is never printed or written.

Sections
  1. Leaderboard: trend following, volatility targeting, mean reversion, calendar effect
  2. Volatility targeting: sensitivity to target / window, transaction costs, rebalance band
  3. Can lagged OHLC predict next-day direction? (logistic regression + MLP, 5 seeds)
  4. Overnight vs intraday decomposition of QQQ's return

These strategies are NOT registered in qqq.strategies.REGISTRY; they are a research
screen, not slot implementations.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qqq import config  # noqa: E402
from qqq import features as F  # noqa: E402
from qqq.backtest import run_backtest  # noqa: E402
from qqq.data import load_prices, split  # noqa: E402
from qqq.evaluate import PERIODS, backtest_strategy  # noqa: E402
from qqq.metrics import bootstrap_sharpe_ci  # noqa: E402
from qqq.strategies.base import BuyAndHold, Strategy  # noqa: E402
from qqq.strategies.moving_average import MovingAverageCrossover  # noqa: E402

DEV = ("train", "validation")
OUT_DIR = config.RESULTS / "research"


# ------------------------------------------------------------- strategies ---
class Rule(Strategy):
    """Wrap a causal function prices -> positions as a Strategy."""

    def __init__(self, name: str, fn):
        self.name, self.fn = name, fn

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        return self.fn(prices).fillna(0.0).clip(-1, 1).astype(float).rename("position")


def _px(p):
    return p["AdjClose"]


def vol_scale(p, target=0.15, window=20, band=0.0):
    """min(1, target / realized vol): no leverage, so it can only reduce exposure.
    band > 0: only rebalance when the target size moves by more than `band`."""
    raw = (target / F.rolling_volatility(p, window)).clip(upper=1.0).fillna(0.0)
    if band <= 0:
        return raw
    out, cur = [], 0.0
    for x in raw.to_numpy():
        if abs(x - cur) > band:
            cur = x
        out.append(cur)
    return pd.Series(out, index=p.index)


def sma_filter(p, window=200):
    """Faber (2007), daily version: long when price > 200-day SMA, else cash."""
    return (_px(p) > F.sma(_px(p), window)).astype(float)


def tsmom(p, lookback):
    """Time-series momentum: long if the trailing `lookback`-day return is positive."""
    m = F.momentum(p, lookback)
    return (m > 0).astype(float).where(m.notna(), 0.0)


def tsmom_blend(p):
    """Hurst, Ooi & Pedersen (2017): equal blend of 1-, 3- and 12-month signals (long/flat)."""
    return sum(tsmom(p, lb) for lb in (21, 63, 252)) / 3


def rsi2(p, entry=10.0, exit_=70.0, trend=200):
    """Connors-style: buy when RSI(2) < entry, exit when RSI(2) > exit_,
    optionally only while price > SMA(trend)."""
    r = F.rsi(_px(p), 2)
    s = pd.Series(np.nan, index=p.index)
    s[r < entry] = 1.0
    s[r > exit_] = 0.0
    pos = s.ffill().fillna(0.0)
    if trend:
        pos[~(_px(p) > F.sma(_px(p), trend))] = 0.0
    return pos


def turn_of_month(p):
    """McConnell & Xu (2008): hold from the last trading day of a month through the
    3rd trading day of the next. position[t] = 1 if the NEXT weekday is in the window.

    Leak-free: the window is computed on a Mon-Fri calendar that is fully known in
    advance, never from the future rows of the price index. (Exchange holidays shift
    the window by a day at most a few times a year.)"""
    out = pd.Series(0.0, index=p.index)
    for t in p.index:
        nxt = t + pd.offsets.BDay(1)
        month_days = pd.bdate_range(nxt.replace(day=1), nxt + pd.offsets.BMonthEnd(0))
        k = month_days.get_loc(nxt)
        if k <= 2 or k == len(month_days) - 1:
            out[t] = 1.0
    return out


def screen_strategies() -> dict[str, Strategy]:
    return {
        "Buy & hold": BuyAndHold(),
        "Slot 1 MA v3 (team default)": MovingAverageCrossover(),
        "Slot 1 MA v2 (no vol cap)": MovingAverageCrossover(vol_target=None),
        "Vol target 15%, always long": Rule("vt15", vol_scale),
        "Trend blend 1/3/12m x vol target": Rule("tsmom_vt",
                                                 lambda p: tsmom_blend(p) * vol_scale(p)),
        "SMA200 x vol target": Rule("sma200_vt", lambda p: sma_filter(p) * vol_scale(p)),
        "Trend blend 1/3/12m": Rule("tsmom_blend", tsmom_blend),
        "SMA200 filter (Faber)": Rule("sma200", sma_filter),
        "12-month momentum": Rule("tsmom12", lambda p: tsmom(p, 252)),
        "RSI(2) + SMA200 filter": Rule("rsi2", rsi2),
        "RSI(2), no filter": Rule("rsi2_nf", lambda p: rsi2(p, trend=None)),
        "Turn of month": Rule("tom", turn_of_month),
    }


# ---------------------------------------------------------------- sections ---
def leaderboard(prices) -> dict[str, pd.DataFrame]:
    rows = {p: {} for p in DEV}
    for label, strat in screen_strategies().items():
        res = backtest_strategy(strat, prices)
        for period in DEV:
            sub = res.slice(*PERIODS[period])
            m = sub.metrics()
            lo, hi = bootstrap_sharpe_ci(sub.returns, rf_daily=sub.rf)  # 2000 resamples, block 20
            rows[period][label] = {
                "CAGR": f"{m['annualized_return']:.1%}",
                "Ann. vol": f"{m['annualized_volatility']:.1%}",
                "Sharpe": f"{m['sharpe_ratio']:.2f}",
                "Sharpe 95% CI": f"[{lo:.2f}, {hi:.2f}]",
                "Max DD": f"{m['max_drawdown']:.1%}",
                "Calmar": f"{m['calmar_ratio']:.2f}",
                "Exposure": f"{m['exposure']:.0%}",
                "# trades": f"{m['n_position_changes']:.0f}",
                "_sharpe": m["sharpe_ratio"],
            }
    out = {}
    for period in DEV:
        df = pd.DataFrame(rows[period]).T
        out[period] = df.sort_values("_sharpe", ascending=False).drop(columns="_sharpe")
    return out


def vol_target_checks(prices) -> tuple[pd.DataFrame, pd.DataFrame]:
    def sharpe(pos, period, cost=config.TRANSACTION_COST_BPS):
        r = run_backtest(prices, pos.rename("position"), cost_bps=cost)
        return r.slice(*PERIODS[period]).metrics()

    grid = {}
    for w in (10, 20, 60):
        grid[f"{w}-day window"] = {
            f"target {t:.0%}": sharpe(vol_scale(prices, t, w), "train")["sharpe_ratio"]
            for t in (0.10, 0.15, 0.20, 0.25)
        }
    grid = pd.DataFrame(grid).round(2)

    costs = {}
    for band in (0.0, 0.10):
        pos = vol_scale(prices, band=band)
        for c in (0, 5, 10, 20):
            tr, va = sharpe(pos, "train", c), sharpe(pos, "validation", c)
            costs[(f"band {band:.0%}", f"{c} bps")] = {
                "Train Sharpe": round(tr["sharpe_ratio"], 2),
                "Validation Sharpe": round(va["sharpe_ratio"], 2),
                "Train turnover / yr": round(tr["annual_turnover"], 1),
            }
    return grid, pd.DataFrame(costs).T


def direction_predictability(prices, lags: int = 10, seeds: int = 5) -> pd.DataFrame:
    """Fit on train, score on validation. Masks are built from forward_return BEFORE
    thresholding, so the last train row (no target) is dropped instead of labelled 0."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    X = F.lagged_ohlc_features(prices, lags=lags)
    fwd = F.forward_return(prices, 1)
    train, val, _ = split(prices)

    def xy(idx):
        ok = X.loc[idx].notna().all(axis=1) & fwd.loc[idx].notna()
        return X.loc[idx][ok], (fwd.loc[idx][ok] > 0).astype(int)

    Xtr, ytr = xy(train.index)
    Xva, yva = xy(val.index)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)

    rows = {}
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr_s, ytr)
    p = lr.predict_proba(Xva_s)[:, 1]
    rows["Logistic regression (L2)"] = {"Validation AUC": f"{roc_auc_score(yva, p):.3f}",
                                        "Share of days predicted up": f"{np.mean(p > 0.5):.0%}"}
    aucs, ups = [], []
    for seed in range(seeds):
        nn = MLPClassifier((32, 16), alpha=1e-3, max_iter=300, random_state=seed).fit(Xtr_s, ytr)
        q = nn.predict_proba(Xva_s)[:, 1]
        aucs.append(roc_auc_score(yva, q))
        ups.append(np.mean(q > 0.5))
    rows[f"MLP (32, 16), {seeds} seeds"] = {
        "Validation AUC": f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
        "Share of days predicted up": f"{np.mean(ups):.0%}"}
    rows["Base rate (validation up days)"] = {"Validation AUC": "0.500",
                                              "Share of days predicted up": f"{yva.mean():.0%}"}
    return pd.DataFrame(rows).T


def overnight_intraday(prices) -> pd.DataFrame:
    train, val, _ = split(prices)
    rows = {}
    for name, d in (("Train", train), ("Validation", val)):
        adj = d["AdjClose"] / d["Close"]
        o, c = d["Open"] * adj, d["AdjClose"]
        overnight = (o / c.shift(1) - 1).dropna()
        intraday = c / o - 1
        rows[name] = {"Overnight (close→open)": f"{(1 + overnight).prod() - 1:.0%}",
                      "Intraday (open→close)": f"{(1 + intraday).prod() - 1:.0%}",
                      "Close→close": f"{c.iloc[-1] / c.iloc[0] - 1:.0%}"}
    return pd.DataFrame(rows).T


# -------------------------------------------------------------------- main ---
def main() -> None:
    warnings.filterwarnings("ignore")
    prices = load_prices(source="csv")
    sections = []

    lb = leaderboard(prices)
    for period in DEV:
        sections.append((f"1. Leaderboard - {period} (sorted by Sharpe)", lb[period]))
    grid, costs = vol_target_checks(prices)
    sections.append(("2a. Vol target: train Sharpe by target and window", grid))
    sections.append(("2b. Vol target: costs and rebalance band", costs))
    sections.append(("3. Next-day direction from 10 days of lagged OHLC",
                     direction_predictability(prices)))
    sections.append(("4. Where QQQ's return was earned (cumulative)", overnight_intraday(prices)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# Strategy screen (train + validation only)\n",
          (f"Costs {config.TRANSACTION_COST_BPS:g} bps one-way; cash earns T-bills; "
           "Sharpe in excess of T-bills. Generated by `scripts/screen_strategies.py`.\n")]
    for title, df in sections:
        print(f"\n=== {title} ===\n{df.to_string()}")
        md.append(f"\n### {title}\n\n{df.to_markdown()}\n")
    (OUT_DIR / "strategy_screen.md").write_text("".join(md))
    print(f"\nWrote {OUT_DIR / 'strategy_screen.md'}")


if __name__ == "__main__":
    main()
