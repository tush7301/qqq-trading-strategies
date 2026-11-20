"""Collation entry point: reproduces every number in the final report.

    python scripts/run_all.py                 # train + validation only
    python scripts/run_all.py --include-test  # final run, ONCE, at collation

For each registered strategy: fit on train -> positions on the full sample
(so indicators have history) -> backtest -> metrics per period -> results/.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from qqq import config
from qqq.backtest import run_backtest
from qqq.data import load_prices, split
from qqq.metrics import compute_metrics
from qqq.strategies import REGISTRY, validate_positions


def main(include_test: bool, source: str) -> None:
    prices = load_prices(source=source)
    train_prices, _, _ = split(prices)
    periods = ["train", "validation"] + (["test"] if include_test else [])
    config.POSITIONS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for name, cls in REGISTRY.items():
        strat = cls().fit(train_prices)
        positions = strat.generate_positions(prices)
        validate_positions(positions, prices)
        positions.rename("position").to_frame().to_csv(config.POSITIONS_DIR / f"{name}.csv")

        result = run_backtest(prices, positions)
        for period, rets in zip(["train", "validation", "test"], split(result.returns)):
            if period not in periods:
                continue
            m = compute_metrics(rets) | {"params": strat.params()}
            (config.METRICS_DIR / f"{name}_{period}.json").write_text(json.dumps(m, indent=2))
        print(f"done: {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--source", default="auto", choices=["auto", "csv", "synthetic"])
    a = ap.parse_args()
    main(a.include_test, a.source)
