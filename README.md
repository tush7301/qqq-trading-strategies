# QQQ Trading Strategies

Final project for **Finance and Structuring for DS**: developing and back-testing trading strategies on **QQQ (Invesco QQQ Trust, Nasdaq-100)**.

The data, backtest engine, metrics, evaluation tools and comparison charts are **done**. Each team member implements **one strategy**. Every strategy runs through the same pipeline, the results are compared on a validation period, and we pick the best one to move ahead with.

**→ Plan, rules and timeline: [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)**

## Strategy slots

| Slot | Strategy | File | Owner | Status |
|---|---|---|---|---|
| — | Buy & hold (benchmark) | `strategies/base.py` | — | ✅ |
| 1 | Moving-average crossover | `strategies/moving_average.py` | _TBD_ | ⬜ |
| 2 | Neural network | `strategies/neural_net.py` | _TBD_ | ⬜ |
| 3 | Mean reversion (RSI/Bollinger) | `strategies/mean_reversion.py` | _TBD_ | ⬜ |
| 4 | Momentum / volatility targeting | `strategies/momentum.py` | _TBD_ | ⬜ |
| 5 | ML classifier (logistic / GBM) | `strategies/ml_classifier.py` | _TBD_ | ⬜ |

## Quick start

```bash
git clone https://github.com/tush7301/qqq-trading-strategies.git && cd qqq-trading-strategies
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -rx                      # infra tests pass; unimplemented slots show as xfail
python scripts/run_all.py       # leaderboard + charts for train & validation → results/
```

Then open `notebooks/00_quickstart.ipynb`. The core loop:

```python
from qqq.evaluate import evaluate, grid_search, check_no_lookahead
from qqq.strategies.moving_average import MovingAverageCrossover

evaluate(MovingAverageCrossover(20, 100))          # train + validation vs buy-and-hold
grid_search(MovingAverageCrossover, {"short_window": [10, 20, 50],
                                     "long_window": [100, 200]})   # ranked on train Sharpe
check_no_lookahead(MovingAverageCrossover())
```

## The three rules
1. `position[t]` may only use data up to the **close of day t**. Don't shift it; the engine trades it from t to t+1.
2. Fit on **train**, choose on **validation**. The **test** period (2022–2026) is opened once, in the final run.
3. Work in your own strategy file and notebook folder, open PRs to `main`, and keep CI green.

## Layout
```
data/raw/QQQ.csv          2005-01-03 → 2026-10-30 daily OHLCV + AdjClose (see data/README.md)
src/qqq/config.py         periods, costs (5 bps), seed, paths
src/qqq/data.py           load_prices(), split()
src/qqq/features.py       causal indicators + lagged OHLC features
src/qqq/backtest.py       run_backtest(), trade_stats()
src/qqq/metrics.py        compute_metrics(), bootstrap CI, t-test
src/qqq/evaluate.py       evaluate(), grid_search(), check_no_lookahead()
src/qqq/reporting.py      leaderboard tables + charts
src/qqq/strategies/       base.py (contract), _template.py, 5 strategy slots
scripts/run_all.py        bake-off: all strategies → results/
tests/                    engine/metric/feature tests + per-strategy contract & look-ahead tests
results/                  generated: positions, metrics/leaderboards, figures
report/                   outline + sections for the final report
docs/                     plan, weekly progress emails
```
