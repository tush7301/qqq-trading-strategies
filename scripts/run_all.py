"""Run every registered strategy and produce the comparison ("bake-off").

    python scripts/run_all.py                  # train + validation (development)
    python scripts/run_all.py --include-test   # FINAL run, once, after the winner is chosen

Outputs (results/):
    positions/<strategy>.csv
    metrics/<strategy>_<period>.json
    metrics/leaderboard_<period>.{csv,md}       sorted by Sharpe
    metrics/cost_sensitivity_validation.{csv,md}
    figures/equity_<period>.png, drawdown_<period>.png, risk_return_<period>.png,
            annual_returns.png
Strategies that still raise NotImplementedError are skipped, not fatal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from qqq import config  # noqa: E402
from qqq.backtest import run_backtest, trade_stats  # noqa: E402
from qqq.data import load_prices, split  # noqa: E402
from qqq.evaluate import PERIODS  # noqa: E402
from qqq.metrics import compute_metrics  # noqa: E402
from qqq.reporting import (comparison_table, plot_annual_returns, plot_drawdowns,  # noqa: E402
                           plot_equity_curves, plot_risk_return, save_table)
from qqq.strategies import REGISTRY, validate_positions  # noqa: E402

COST_GRID = (0.0, 5.0, 10.0, 20.0)


def main(include_test: bool, source: str) -> None:
    prices = load_prices(source=source)
    train, _, _ = split(prices)
    periods = ["train", "validation"] + (["test"] if include_test else [])
    for d in (config.POSITIONS_DIR, config.METRICS_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    positions, results, skipped = {}, {}, []
    for name, cls in REGISTRY.items():
        try:
            strat = cls().fit(train)
            pos = strat.generate_positions(prices)
        except NotImplementedError:
            skipped.append(name)
            continue
        validate_positions(pos, prices)
        positions[name] = (strat, pos)
        results[name] = run_backtest(prices, pos)
        pos.rename("position").to_frame().to_csv(config.POSITIONS_DIR / f"{name}.csv")
        print(f"  ran      {name:<20} {strat.params()}")
    for name in skipped:
        print(f"  skipped  {name:<20} (not implemented yet)")

    for period in periods:
        start, end = PERIODS[period]
        table = {}
        for name, res in results.items():
            sub = res.slice(start, end)
            m = compute_metrics(sub.returns) | trade_stats(sub)
            table[name] = m
            payload = m | {"params": positions[name][0].params()}
            (config.METRICS_DIR / f"{name}_{period}.json").write_text(json.dumps(payload, indent=2))
        lb = comparison_table(table)
        save_table(lb, config.METRICS_DIR / f"leaderboard_{period}", f"Leaderboard - {period}")
        rets = {n: r.slice(start, end).returns for n, r in results.items()}
        plot_equity_curves(rets, f"Growth of $1 - {period}", config.FIGURES_DIR / f"equity_{period}.png")
        plot_drawdowns(rets, f"Drawdowns - {period}", config.FIGURES_DIR / f"drawdown_{period}.png")
        plot_risk_return(lb, f"Risk vs return - {period}", config.FIGURES_DIR / f"risk_return_{period}.png")
        print(f"\n=== {period} ===")
        print(lb[["annualized_return", "annualized_volatility", "sharpe_ratio",
                  "max_drawdown", "cumulative_return"]].round(3).to_string())

    # Full-period curve (test shaded only if included).
    last = PERIODS[periods[-1]][1]
    full = {n: r.slice(None, last).returns for n, r in results.items()}
    shade = (config.VALIDATION_START, config.VALIDATION_END)
    plot_equity_curves(full, "Growth of $1 - all development periods (validation shaded)",
                       config.FIGURES_DIR / "equity_all.png", shade=shade)
    plot_annual_returns(full, "Calendar-year returns", config.FIGURES_DIR / "annual_returns.png")

    # Transaction-cost sensitivity on validation.
    v0, v1 = PERIODS["validation"]
    rows = {}
    for name, (_, pos) in positions.items():
        rows[name] = {f"{c:g} bps": compute_metrics(
            run_backtest(prices, pos, cost_bps=c).slice(v0, v1).returns)["sharpe_ratio"]
            for c in COST_GRID}
    cs = pd.DataFrame(rows).T
    cs.to_csv(config.METRICS_DIR / "cost_sensitivity_validation.csv")
    (config.METRICS_DIR / "cost_sensitivity_validation.md").write_text(
        "### Validation Sharpe vs one-way transaction cost\n\n" + cs.round(2).to_markdown() + "\n")
    print(f"\nWrote results to {config.RESULTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--source", default="auto", choices=["auto", "csv", "synthetic"])
    a = ap.parse_args()
    main(a.include_test, a.source)
