"""Export a strategy configuration's results into the repo (committed, for the report).

    python scripts/export_results.py ma_crossover tuned
    python scripts/export_results.py ma_crossover v1 --params '{"long_window": 200, "reentry_window": null}'

Writes results/strategies/<strategy>/<label>/:
    params.json                      configuration used
    metrics.csv / metrics.md         train + validation metrics vs buy-and-hold
    metrics_<period>.json            full metric dicts
    annual_returns.csv               calendar-year returns vs buy-and-hold
    positions.csv                    daily positions (train + validation only)
    equity.png, drawdown.png, annual_returns.png

Only development periods are exported. Test-period results come from
`scripts/run_all.py --include-test` at the final run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from qqq import config  # noqa: E402
from qqq.data import load_prices  # noqa: E402
from qqq.evaluate import DEV_PERIODS, PERIODS, backtest_strategy  # noqa: E402
from qqq.metrics import annual_returns  # noqa: E402
from qqq.reporting import (format_table, plot_annual_returns, plot_drawdowns,  # noqa: E402
                           plot_equity_curves)
from qqq.strategies import REGISTRY, BuyAndHold  # noqa: E402

COLS = ["mean_daily_return", "std_daily_return", "annualized_mean_return", "annualized_return",
        "annualized_volatility", "sharpe_ratio", "sortino_ratio", "cumulative_return",
        "max_drawdown", "calmar_ratio", "hit_rate", "exposure", "n_position_changes",
        "annual_turnover"]


def export(strategy_name: str, label: str, params: dict) -> Path:
    out = config.RESULTS / "strategies" / strategy_name / label
    out.mkdir(parents=True, exist_ok=True)
    prices = load_prices(source="csv")
    strat = REGISTRY[strategy_name](**params)
    res = backtest_strategy(strat, prices)
    bh = backtest_strategy(BuyAndHold(), prices)
    end = PERIODS[DEV_PERIODS[-1]][1]

    (out / "params.json").write_text(json.dumps(strat.params(), indent=2) + "\n")

    rows = {}
    for period in DEV_PERIODS:
        for name, r in ((label, res), ("buy_and_hold", bh)):
            m = r.slice(*PERIODS[period]).metrics()
            rows[(name, period)] = m
            if name == label:
                (out / f"metrics_{period}.json").write_text(json.dumps(m, indent=2) + "\n")
    table = pd.DataFrame(rows).T[COLS]
    table.index.names = ["strategy", "period"]
    table.to_csv(out / "metrics.csv", float_format="%.6f")
    flat = table.astype(float)
    flat.index = [f"{s} · {p}" for s, p in flat.index]
    md = format_table(flat).to_markdown()
    (out / "metrics.md").write_text(
        f"### {strategy_name} - {label}\n\nParams: `{json.dumps(strat.params())}`  \n"
        f"Costs: {config.TRANSACTION_COST_BPS:g} bps one-way. Periods: train <= {config.TRAIN_END}, "
        f"validation {config.VALIDATION_START} .. {config.VALIDATION_END}.\n\n{md}\n")

    dev = {label: res.slice(None, end).returns, "buy_and_hold": bh.slice(None, end).returns}
    pd.DataFrame({k: annual_returns(v) for k, v in dev.items()}).rename_axis("year").to_csv(
        out / "annual_returns.csv", float_format="%.6f")
    res.positions.loc[:end].rename("position").to_frame().to_csv(out / "positions.csv")

    shade = (config.VALIDATION_START, config.VALIDATION_END)
    plot_equity_curves(dev, f"{strategy_name} ({label}) - growth of $1, validation shaded",
                       out / "equity.png", shade=shade)
    plot_drawdowns(dev, f"{strategy_name} ({label}) - drawdowns", out / "drawdown.png")
    plot_annual_returns(dev, f"{strategy_name} ({label}) - calendar-year returns",
                        out / "annual_returns.png")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", choices=[k for k in REGISTRY if k != "buy_and_hold"])
    ap.add_argument("label", help="folder name, e.g. v1 or tuned")
    ap.add_argument("--params", default="{}", help="JSON kwargs for the strategy constructor")
    a = ap.parse_args()
    print("wrote", export(a.strategy, a.label, json.loads(a.params)))
