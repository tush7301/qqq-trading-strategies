# QQQ Trading Strategies

Final project for **Finance and Structuring for DS**: developing and back-testing trading strategies on **QQQ (Invesco QQQ Trust, Nasdaq-100)**.

The data, backtest engine, metrics, evaluation tools and comparison charts are **done**. Each team member implements **one strategy**. Every strategy runs through the same pipeline, the results are compared on a validation period, and we pick the best one to move ahead with.

**→ Plan, rules and timeline: [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)**

## Strategy slots

| Slot | Strategy | Owner | Issue | Build | Tune | Results | Report |
|---|---|---|---|---|---|---|---|
| — | Buy & hold (benchmark) | — | — | ✅ | — | — | — |
| 1 | [Moving-average crossover](src/qqq/strategies/moving_average.py) | tush7301 | [#3](https://github.com/tush7301/qqq-trading-strategies/issues/3) | ✅ | ✅ [notebook](notebooks/01_moving_average/tuning.ipynb) | ✅ [v1–v3](results/strategies/ma_crossover/README.md) | ✅ [§4.1](report/sections/04_1_moving_average.md) |
| 2 | [Neural network](src/qqq/strategies/neural_net.py) | akshatbhandari15 | [#4](https://github.com/tush7301/qqq-trading-strategies/issues/4) | ✅ | ✅ [protocol](notebooks/02_neural_net/PROTOCOL.md) · [notebook](notebooks/02_neural_net/tuning.ipynb) | ✅ [v1–v2b](results/strategies/neural_net/README.md) | ✅ [§4.2](report/sections/04_2_neural_network.md) |
| 3 | [Mean reversion (RSI / Bollinger)](src/qqq/strategies/mean_reversion.py) | _unassigned_ | [#7](https://github.com/tush7301/qqq-trading-strategies/issues/7) | ⬜ | ⬜ | ⬜ | ⬜ |
| 4 | [Momentum / volatility targeting](src/qqq/strategies/momentum.py) | _unassigned_ | [#8](https://github.com/tush7301/qqq-trading-strategies/issues/8) | ⬜ | ⬜ | ⬜ | ⬜ |
| 5 | [ML classifier (logistic / GBM)](src/qqq/strategies/ml_classifier.py) | _unassigned_ | [#9](https://github.com/tush7301/qqq-trading-strategies/issues/9) | ⬜ | ⬜ | ⬜ | ⬜ |

✅ done · 🟡 in progress · ⬜ not started. **Build** = implemented, tests green, merged. **Tune** = parameters chosen on train/validation and set as defaults. **Results** = exported with `scripts/export_results.py`. **Report** = section in `report/sections/`.

### Current standings (development periods only)
Net of 5 bps costs; cash earns T-bills; Sharpe in excess of T-bills. The test period (2022 →) is sealed until the final run.

| Strategy | Train Sharpe | Train max DD | Validation Sharpe | Validation max DD |
|---|---|---|---|---|
| Slot 1: MA crossover v3 (SMA 50/250 + re-entry + vol cap) | **0.76** | −26.8% | 1.44 | −19.0% |
| Slot 2: Neural network v2b (Sharpe-trained MLP, vol-rescaled)¹ | 0.72 | −39.2% | **1.53** | **−16.2%** |
| Buy & hold | 0.56 | −53.4% | 1.38 | −28.6% |

¹ Train figures of a fitted network are in-sample. Its out-of-fold walk-forward Sharpe on 2011–2018 is 0.89, the same as buy-and-hold's. The vol rescaling was added post-hoc (see the Slot 2 protocol).

> **Note for Slot 4:** Slot 1 already uses volatility targeting as a sizing overlay. To keep the comparison between *different techniques* meaningful, Slot 4 should focus on the **momentum** signal (e.g. 3–12-month time-series momentum), with or without its own sizing. Combining winners is planned after the selection meeting (Plan §4).

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
