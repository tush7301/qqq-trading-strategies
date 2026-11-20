# QQQ Trading Strategies

Final project for **Finance and Structuring for DS**: developing and back-testing trading strategies on **QQQ (Invesco QQQ Trust, Nasdaq-100)**.

We build two or more strategies (moving-average crossover and a neural network) and backtest them with no look-ahead bias and a train / validation / test split. For each one we estimate the expected return and standard deviation (daily and annualized), cumulative return and maximum drawdown, and compare against buy-and-hold.

**→ Read [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) first.** It defines the 5 workstreams, the frozen interfaces between them, and the timeline.

## Workstreams

| WS | Area | Owner | Main files |
|---|---|---|---|
| 1 | Data & Features | _TBD_ | `src/qqq/data.py`, `src/qqq/features.py`, `data/` |
| 2 | Backtest Engine & Metrics | _TBD_ | `src/qqq/backtest.py`, `src/qqq/metrics.py` |
| 3 | Moving Average Strategy | _TBD_ | `src/qqq/strategies/moving_average.py` |
| 4 | Neural Network Strategy | _TBD_ | `src/qqq/strategies/neural_net.py` |
| 5 | Evaluation, Report & Team Rep | _TBD_ | `src/qqq/reporting.py`, `report/` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,nn for PyTorch
pytest -rx                       # contract tests; unimplemented parts show as xfail
python scripts/run_all.py --source synthetic   # end-to-end once WS2+ are in
```

Until real data is committed, `load_prices(source="synthetic")` returns fake data in the exact same format, so nobody is blocked.

## Layout

```
src/qqq/            library code (config, data, features, backtest, metrics, strategies, reporting)
scripts/run_all.py  reproduces every number in the report
tests/              contract + look-ahead tests (CI runs these on every PR)
notebooks/wsN_*/    exploration per workstream
data/raw/           QQQ.csv (committed; part of the submission)
results/            positions, metrics JSON, figures
report/             final report sources
docs/               plan + weekly progress reports
```

## Rules of the road
1. Strategies return positions for day *t* using data up to the close of *t*. **Only the engine shifts them.**
2. Fit only on train. Choose parameters on validation. **The test set is run once, on Dec 14.**
3. Work in your own files, open PRs to `main`, and keep CI green.
