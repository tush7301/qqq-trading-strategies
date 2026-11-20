"""WS5 - Comparison tables and figures. Owner: <WS5 owner>

Input contract (produced by every strategy owner, or by scripts/run_all.py):
    results/positions/<strategy>.csv   columns: Date, position
    results/metrics/<strategy>_<period>.json   keys = qqq.metrics.METRIC_KEYS + "params"
    period in {"train", "validation", "test"}
"""
from __future__ import annotations

import pandas as pd


def comparison_table(metrics_by_strategy: dict[str, dict]) -> pd.DataFrame:
    """WS5 TODO: rows = strategies (buy_and_hold first), cols = metrics,
    formatted as % where relevant. Export to results/metrics/comparison_<period>.csv/.md."""
    raise NotImplementedError("WS5")


def plot_equity_curves(equity_by_strategy: dict[str, pd.Series], title: str, path) -> None:
    """WS5 TODO: log-scale growth of $1, test-period shading."""
    raise NotImplementedError("WS5")


def plot_drawdowns(equity_by_strategy: dict[str, pd.Series], path) -> None:
    """WS5 TODO: underwater chart."""
    raise NotImplementedError("WS5")


def plot_risk_return(metrics_by_strategy: dict[str, dict], path) -> None:
    """WS5 TODO: annualized volatility (x) vs annualized return (y) scatter."""
    raise NotImplementedError("WS5")
