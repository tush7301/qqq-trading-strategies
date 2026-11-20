"""Performance metrics (infrastructure - done).

All metrics take a series of DAILY simple returns.

    mean_daily_return       mean(r)                         <- "expected return" in the brief
    std_daily_return        std(r, ddof=1)                  <- "standard deviation"
    annualized_mean_return  mean(r) * 252                   (arithmetic)
    annualized_return       prod(1+r)^(252/n) - 1           (geometric / CAGR)
    annualized_volatility   std(r) * sqrt(252)
    sharpe_ratio            mean(r - rf) / std(r - rf) * sqrt(252)   (excess of T-bills)
    sortino_ratio           mean(r - rf) / downside_std(r - rf) * sqrt(252)

rf is the daily T-bill series (rf_daily, from BacktestResult.rf) when given,
else the constant config.RISK_FREE_RATE_ANNUAL. Use BacktestResult.metrics()
so the right series is always passed.
    cumulative_return       prod(1+r) - 1
    max_drawdown            min_t equity_t / max_{s<=t} equity_s - 1   (<= 0)
    calmar_ratio            annualized_return / |max_drawdown|
    hit_rate                share of r > 0 among days with r != 0
    n_days                  len(r)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qqq import config

METRIC_KEYS = [
    "mean_daily_return", "std_daily_return", "annualized_mean_return", "annualized_return",
    "annualized_volatility", "sharpe_ratio", "sortino_ratio", "cumulative_return",
    "max_drawdown", "calmar_ratio", "hit_rate", "n_days",
]


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b and np.isfinite(b) and b != 0 else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall of an equity curve. Starting capital (1.0)
    counts as a peak, so a loss on day 1 is a drawdown."""
    if equity.empty:
        return float("nan")
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    return float((equity.to_numpy() / peak - 1).min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = (1 + returns).cumprod()
    peak = eq.cummax().clip(lower=1.0)
    return (eq / peak - 1).rename("drawdown")


def compute_metrics(
    returns: pd.Series,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
    rf_annual: float = config.RISK_FREE_RATE_ANNUAL,
    rf_daily: pd.Series | None = None,
) -> dict:
    r = returns.dropna().astype("float64")
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in METRIC_KEYS} | {"n_days": n}
    if rf_daily is not None:
        rf_d = rf_daily.reindex(r.index).fillna(0.0)
    else:
        rf_d = pd.Series((1 + rf_annual) ** (1 / periods_per_year) - 1, index=r.index)
    ex = r - rf_d
    mean, std = float(r.mean()), float(r.std(ddof=1))
    ex_mean, ex_std = float(ex.mean()), float(ex.std(ddof=1))
    downside = float(np.sqrt((np.minimum(ex, 0) ** 2).mean()))
    growth = float((1 + r).prod())
    ann_ret = growth ** (periods_per_year / n) - 1 if growth > 0 else -1.0
    mdd = max_drawdown((1 + r).cumprod())
    active = r[r != 0]
    return {
        "mean_daily_return": mean,
        "std_daily_return": std,
        "annualized_mean_return": mean * periods_per_year,
        "annualized_return": ann_ret,
        "annualized_volatility": std * np.sqrt(periods_per_year),
        "sharpe_ratio": _safe_div(ex_mean, ex_std) * np.sqrt(periods_per_year),
        "sortino_ratio": _safe_div(ex_mean, downside) * np.sqrt(periods_per_year),
        "cumulative_return": growth - 1,
        "max_drawdown": mdd,
        "calmar_ratio": _safe_div(ann_ret, abs(mdd)),
        "hit_rate": float((active > 0).mean()) if len(active) else float("nan"),
        "n_days": n,
    }


def annual_returns(returns: pd.Series) -> pd.Series:
    """Calendar-year compounded returns."""
    return (1 + returns).groupby(returns.index.year).prod() - 1


def excess_return_test(strategy: pd.Series, benchmark: pd.Series) -> dict:
    """Paired t-test (Newey-West-free, simple) on daily return differences.
    Use as a rough guide only - daily returns are autocorrelated/heteroskedastic."""
    from scipy import stats

    d = (strategy - benchmark).dropna()
    t, p = stats.ttest_1samp(d, 0.0)
    return {"mean_daily_diff": float(d.mean()), "t_stat": float(t), "p_value": float(p)}


def bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 2000, block: int = 20,
                        alpha: float = 0.05, seed: int = config.RANDOM_SEED,
                        rf_daily: pd.Series | None = None) -> tuple[float, float]:
    """Moving-block bootstrap CI for the annualized Sharpe ratio (of excess returns
    when rf_daily is given)."""
    rr = returns.dropna()
    if rf_daily is not None:
        rr = rr - rf_daily.reindex(rr.index).fillna(0.0)
    r = rr.to_numpy()
    n = len(r)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        s = np.concatenate([r[j:j + block] for j in starts])[:n]
        out[i] = s.mean() / s.std(ddof=1) * np.sqrt(252) if s.std() > 0 else np.nan
    lo, hi = np.nanquantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
