"""Comparison tables and figures (infrastructure - done)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from qqq.metrics import drawdown_series  # noqa: E402

PCT_COLS = ["mean_daily_return", "std_daily_return", "annualized_mean_return", "annualized_return",
            "annualized_volatility", "cumulative_return", "max_drawdown", "hit_rate", "exposure",
            "pct_days_long", "pct_days_short"]
SHOW_COLS = ["mean_daily_return", "std_daily_return", "annualized_mean_return",
             "annualized_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
             "cumulative_return", "max_drawdown", "calmar_ratio", "hit_rate", "exposure",
             "n_position_changes", "annual_turnover"]
LABELS = {
    "mean_daily_return": "Mean daily ret", "std_daily_return": "Std daily ret",
    "annualized_mean_return": "Ann. mean ret", "annualized_return": "CAGR",
    "annualized_volatility": "Ann. vol", "sharpe_ratio": "Sharpe", "sortino_ratio": "Sortino",
    "cumulative_return": "Cum. ret", "max_drawdown": "Max DD", "calmar_ratio": "Calmar",
    "hit_rate": "Hit rate", "exposure": "Exposure", "n_position_changes": "# trades",
    "annual_turnover": "Turnover/yr",
}


def comparison_table(metrics_by_strategy: dict[str, dict], sort_by: str = "sharpe_ratio") -> pd.DataFrame:
    """Rows = strategies (sorted by `sort_by`, buy_and_hold always shown), cols = metrics."""
    df = pd.DataFrame(metrics_by_strategy).T
    df = df[[c for c in SHOW_COLS if c in df]].astype(float)
    return df.sort_values(sort_by, ascending=False)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in df:
        if c in PCT_COLS:
            digits = 3 if "daily" in c else 1
            out[c] = df[c].map(lambda v: f"{v:.{digits}%}")
        elif c in ("n_position_changes",):
            out[c] = df[c].map(lambda v: f"{v:.0f}")
        else:
            out[c] = df[c].map(lambda v: f"{v:.2f}")
    return out.rename(columns=LABELS)


def save_table(df: pd.DataFrame, path_stem: Path, title: str = "") -> None:
    df.to_csv(path_stem.with_suffix(".csv"))
    md = format_table(df).to_markdown()
    path_stem.with_suffix(".md").write_text((f"### {title}\n\n" if title else "") + md + "\n")


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def _color(name: str, i: int):
    return "black" if name == "buy_and_hold" else plt.cm.tab10(i % 10)


def plot_equity_curves(returns_by_strategy: dict[str, pd.Series], title: str, path: Path,
                       shade: tuple | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, r) in enumerate(returns_by_strategy.items()):
        lw, ls = (2.0, "--") if name == "buy_and_hold" else (1.4, "-")
        ax.plot((1 + r).cumprod(), label=name, color=_color(name, i), lw=lw, ls=ls)
    if shade:
        ax.axvspan(pd.Timestamp(shade[0]), pd.Timestamp(shade[1]), color="grey", alpha=0.12)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drawdowns(returns_by_strategy: dict[str, pd.Series], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (name, r) in enumerate(returns_by_strategy.items()):
        ax.plot(drawdown_series(r), label=name, color=_color(name, i), lw=1.2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9)
    _style(ax, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_return(table: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, (name, row) in enumerate(table.iterrows()):
        ax.scatter(row["annualized_volatility"], row["annualized_return"], s=60,
                   color=_color(name, i), marker="s" if name == "buy_and_hold" else "o")
        ax.annotate(name, (row["annualized_volatility"], row["annualized_return"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return (CAGR)")
    _style(ax, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_annual_returns(returns_by_strategy: dict[str, pd.Series], title: str, path: Path) -> None:
    from qqq.metrics import annual_returns

    df = pd.DataFrame({k: annual_returns(v) for k, v in returns_by_strategy.items()})
    fig, ax = plt.subplots(figsize=(10, 4))
    df.plot.bar(ax=ax, width=0.8, color=[_color(n, i) for i, n in enumerate(df)])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8, ncol=3)
    _style(ax, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
