# Implementation Plan: QQQ Trading Strategies

**Course:** Finance and Structuring for DS, Final Project
**Asset:** QQQ (Invesco QQQ Trust, Nasdaq-100)
**Due:** December 22 (internal target: **Sunday Dec 20**, leaving 2 days of buffer)
**Weekly progress email:** every Sunday, sent by the team representative (see §7)

---

## 1. What the brief requires (acceptance checklist)

| # | Requirement | Owner |
|---|---|---|
| R1 | ≥ 2 trading strategies built with *different* techniques | WS3 (moving average), WS4 (neural net) |
| R2 | Implemented in Python and backtested on historical data | WS2 + all |
| R3 | Expected (mean) return and std dev of returns, **daily and annualized** | WS2 |
| R4 | Cumulative return and maximum drawdown | WS2 |
| R5 | Comparison against a **buy-and-hold** benchmark | WS2 (benchmark) + WS5 (comparison) |
| R6 | No look-ahead bias | Everyone. Enforced by the contract and tests (§3) |
| R7 | Separate training and testing periods (out-of-sample) | WS1 (splits) + WS3/WS4 |
| R8 | Report: problem, methods, data, backtest procedure, results, strengths/weaknesses/limitations | WS5 (editor), each WS writes its own section |
| R9 | Historical data + documented, reproducible code submitted | WS1 (data) + WS5 (`run_all.py` reproducibility check) |

---

## 2. Architecture

```
            ┌──────────── WS1 ────────────┐
            │ data/raw/QQQ.csv            │
            │ qqq.data.load_prices()      │──┐
            │ qqq.features.*              │  │  prices DataFrame (contract §3.1)
            └─────────────────────────────┘  │
                                             ▼
   ┌──── WS3 ────┐   ┌──── WS4 ────┐   ┌─ benchmark ─┐
   │ MA crossover │   │ Neural net  │   │ Buy & hold  │   Strategy.fit(train)
   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   Strategy.generate_positions(prices)
          └────────── positions Series (contract §3.2) ──┘
                                             ▼
            ┌──────────── WS2 ────────────┐
            │ qqq.backtest.run_backtest() │  daily net returns, equity, turnover
            │ qqq.metrics.compute_metrics │  dict of metrics (contract §3.4)
            └─────────────────────────────┘
                                             ▼
            ┌──────────── WS5 ────────────┐
            │ qqq.reporting.* + report/   │  tables, charts, final PDF
            └─────────────────────────────┘
```

`scripts/run_all.py` connects everything at collation. Because each arrow is a
**frozen interface that already exists as code in this repo**, no workstream has
to wait for another:

- Until WS1 delivers real data, everyone develops on `load_prices(source="synthetic")`, which returns data in the same shape.
- Until WS2 delivers the engine, WS3 and WS4 can sanity-check with a 3-line local backtest: `(pos.shift(1) * ret).add(1).cumprod()`.
- Until WS3 and WS4 deliver, WS2 and WS5 build and test on `BuyAndHold` and random positions.

---

## 3. Interface contracts (frozen at kickoff)

Changes need a PR approved by **all five** owners. Code lives in `src/qqq/`.

### 3.1 Prices (`qqq.data.load_prices`)
- `DatetimeIndex` named `Date`: trading days only, ascending, unique
- Columns `Open, High, Low, Close, AdjClose, Volume` (float64), with no NaNs
- **Returns are always computed on `AdjClose`**

### 3.2 Strategy (`qqq.strategies.base.Strategy`)
- `fit(train_prices)`: may only look at the training period
- `generate_positions(prices) -> pd.Series`: same index as `prices`, values in `[-1, 1]`, warm-up = 0
- `params() -> dict`: hyper-parameters, for logging
- Register the class in `qqq/strategies/__init__.py::REGISTRY`

### 3.3 Timing rule (anti-look-ahead)
`position[t]` uses information **up to the close of day t** and is held from close t to close t+1.
**Only the engine shifts:** `net[t] = position[t-1] * ret[t] - cost`. Strategies never shift their own positions.
`tests/test_contracts.py::test_strategy_has_no_lookahead` checks this automatically: positions must not change when future data is removed.

### 3.4 Metrics (`qqq.metrics.METRIC_KEYS`)
`mean_daily_return, std_daily_return, annualized_return (geometric), annualized_mean_return (×252), annualized_volatility (×√252), sharpe_ratio, cumulative_return, max_drawdown, calmar_ratio, hit_rate, n_days`

### 3.5 Periods (`qqq.config`)
| Period | Dates | Use |
|---|---|---|
| Train | 2005-01-01 → 2018-12-31 | fit models, initial parameter search |
| Validation | 2019-01-01 → 2021-12-31 | choose between candidates (windows, architectures, thresholds) |
| Test | 2022-01-01 → 2026-10-30 | **run once at collation**; these are the headline numbers |

The test period includes the 2022 bear market (QQQ fell about 33%) and the 2023–24 AI rally, so it covers both a downturn and a strong uptrend.
Costs: 5 bps one-way per unit of turnover (WS5 runs 0/5/10/20 bps sensitivity). Risk-free rate: 0 (WS2 may propose T-bills).

### 3.6 Result files
- `results/positions/<strategy>.csv`: `Date, position`
- `results/metrics/<strategy>_<period>.json`: metric keys + `params`
- `results/figures/*.png`

---

## 4. Workstreams

Each workstream owns its files. The only shared files are `config.py`, `strategies/__init__.py` (one line per strategy), and `report/`.

### WS1: Data & Features
**Owns:** `src/qqq/data.py`, `src/qqq/features.py`, `data/`, `notebooks/ws1_data/`

| Deliverable | Detail | Due |
|---|---|---|
| `download_prices()` | yfinance `QQQ`, `auto_adjust=False`, rename `Adj Close`→`AdjClose`, save `data/raw/QQQ.csv` | **Sun Nov 22** |
| `data/README.md` | Source, download date, row count, adjustment method, known issues | Nov 22 |
| Data quality EDA notebook | Missing days, splits (QQQ 2:1 in 2000 is before our window), outliers, return distribution, summary statistics per period (for the report's Data section) | Nov 29 |
| `lagged_ohlc_features()` | Scale-free lags: log return, (H−L)/C, (C−O)/O, overnight gap, volume z-score (rolling, past only) | Nov 29 |
| `technical_indicators()` | SMA/EMA, RSI, MACD, Bollinger %B, rolling vol. Unit tests for each | Dec 6 |
| *Stretch:* 3rd strategy | e.g. RSI/Bollinger mean-reversion or volatility-targeted trend. Adds a third technique for the comparison | Dec 10 |
| Report section | "Historical Data" (½–1 page) | Dec 13 |

**Done when:** `test_real_prices_match_contract` passes in CI; features pass a truncation (no-look-ahead) test.

### WS2: Backtest Engine & Metrics
**Owns:** `src/qqq/backtest.py`, `src/qqq/metrics.py`, `notebooks/ws2_backtest/`

| Deliverable | Detail | Due |
|---|---|---|
| `run_backtest()` | Timing per §3.3, transaction costs on turnover, returns/equity/turnover | **Nov 25** |
| `compute_metrics()` + `max_drawdown()` | All keys in §3.4. Hand-checked against a small spreadsheet example | Nov 25 |
| Benchmark | Buy-and-hold metrics for all three periods, posted for the team | Nov 29 |
| Extra analytics | Number of trades, average holding period, exposure %, annual returns table, rolling 1-year Sharpe | Dec 6 |
| Statistical significance | Bootstrap CI for mean return / Sharpe, and/or t-test of strategy − benchmark daily returns | Dec 10 |
| Report section | "Back-testing Procedure" (timing, costs, metric definitions with formulas) | Dec 13 |

**Done when:** `test_backtest_timing` and `test_metrics_keys` pass; metrics match a hand calculation to 1e-10.

### WS3: Moving Average Strategy
**Owns:** `src/qqq/strategies/moving_average.py`, `notebooks/ws3_moving_average/`

| Deliverable | Detail | Due |
|---|---|---|
| `MovingAverageCrossover` | SMA short/long crossover; long/flat by default, optional long/short | **Nov 25** |
| Parameter grid on **train** | short ∈ {5,10,20,50}, long ∈ {50,100,150,200}; heat-map of Sharpe | Dec 2 |
| Select on **validation** | Pick the final (short, long) without looking at test; document why | Dec 6 |
| Variants (optional) | EMA vs SMA, a "buffer" band to reduce whipsaw | Dec 6 |
| Report section | Methodology, chosen parameters, heat-map, strengths/weaknesses (lags at turning points, whipsaw in sideways markets) | Dec 13 |

**Done when:** contract + look-ahead tests pass; final params committed as class defaults.

### WS4: Neural Network Strategy
**Owns:** `src/qqq/strategies/neural_net.py`, `notebooks/ws4_neural_net/`

| Deliverable | Detail | Due |
|---|---|---|
| Framework decision | sklearn `MLPClassifier` (simplest, reproducible) vs PyTorch (if trying LSTM). Recommend starting with sklearn | Nov 22 |
| Baseline model | Inputs: last N days of OHLC-derived features. Target: 1{r_{t+1}>0}. Scaler fitted on train only | Nov 29 |
| Validation tuning | lags ∈ {5,10,20}, hidden sizes, regularization, probability threshold; report accuracy/AUC **and** strategy Sharpe | Dec 6 |
| Robustness | Several random seeds (report mean ± sd); optional walk-forward retraining (yearly refit using only past data) | Dec 10 |
| Report section | Architecture, features, training procedure, overfitting checks, limitations (non-stationarity, low signal-to-noise) | Dec 13 |

**Done when:** contract + look-ahead tests pass; results reproducible with a fixed seed; the model is fit inside `fit()` only.

### WS5: Evaluation, Visualization & Report (+ team representative)
**Owns:** `src/qqq/reporting.py`, `report/`, `notebooks/ws5_report/`, `docs/progress_reports/`

| Deliverable | Detail | Due |
|---|---|---|
| Weekly progress emails | Template in `docs/progress_reports/`. Collect a 2-line update from each WS by Saturday night | Every Sunday |
| Report skeleton | `report/REPORT_OUTLINE.md` with page budget per section; intro + QQQ background (Nasdaq-100, tech concentration, 2005–2026 history) | Nov 29 |
| Plotting + tables | `comparison_table`, equity curves (log), drawdown chart, risk-return scatter, annual-returns bar chart. Built on BuyAndHold + random positions so it is ready before the strategies are | Dec 6 |
| Robustness analysis | Transaction-cost sensitivity (0/5/10/20 bps), sub-period results (2022 bear vs 2023–26 bull) | Dec 13 |
| Collation + final report | Run `run_all.py --include-test`, fill in results, write Results & Conclusions, edit for one voice, export PDF | Dec 20 |
| Reproducibility check | Fresh clone → `pip install -e .` → `run_all.py` reproduces every number in the report | Dec 20 |

**Done when:** a single command regenerates every table and figure in the report.

---

## 5. Timeline

| Week | Dates | Milestone |
|---|---|---|
| 0 | Thu Nov 19 – Sun Nov 22 | Kickoff, assign owners, freeze contracts. **WS1 ships real data.** WS4 picks a framework. Progress email #1 |
| 1 | Nov 23 – Nov 29 | First versions of the engine (WS2), MA strategy (WS3) and NN baseline (WS4); features (WS1); report skeleton (WS5). *(Thanksgiving Nov 26: plan around it)* Email #2 |
| 2 | Nov 30 – Dec 6 | Tuning on validation. Plotting ready. **Integration checkpoint Sun Dec 6:** `run_all.py` runs end-to-end on train + validation. Email #3 |
| 3 | Dec 7 – Dec 13 | Robustness, significance tests, stretch strategy. **All report sections drafted by Sun Dec 13.** Parameters frozen. Email #4 |
| 4 | Dec 14 – Dec 20 | **Run the test set once** (Dec 14). Results, conclusions, editing, reproducibility check. Final internal review Sat Dec 19. Email #5 |
| — | Dec 21 – 22 | Buffer → **submit** |

> The brief says "Sunday, December 22nd", but in 2026 Dec 22 is a Tuesday. Confirm the exact deadline with the instructor. The plan targets Sunday Dec 20 either way.

---

## 6. Working agreements

- **Branches:** `ws1-data`, `ws2-backtest`, `ws3-ma`, `ws4-nn`, `ws5-report`. Open a PR to `main` and get 1 reviewer from a *different* workstream. CI (`pytest`) must be green.
- **Stay in your own files.** Touching another workstream's file means opening a PR that its owner reviews.
- **Notebooks** are for exploration and figures. Logic the final pipeline needs goes into `src/qqq/`. Clear notebook outputs before committing, except final figures.
- **No peeking at the test set.** Nobody computes test-period metrics before Dec 14. `run_all.py` excludes the test period unless `--include-test` is passed.
- **Seeds:** use `config.RANDOM_SEED` everywhere.
- **Report sections:** each owner writes theirs in Markdown under `report/sections/`, and WS5 merges them.

## 7. Weekly progress emails (team representative = WS5 owner)

Due Sundays Nov 22, Nov 29, Dec 6, Dec 13 and Dec 20. Save a copy of each in `docs/progress_reports/YYYY-MM-DD.md`. Template: `docs/progress_reports/TEMPLATE.md`.

## 8. Risks

| Risk | Mitigation |
|---|---|
| NN overfits or makes no money | That is a valid finding. Report it honestly, with validation-vs-test comparison and seed variance. Signal-to-noise in daily returns is low |
| Hidden look-ahead (e.g. scaler fitted on all data, centered rolling windows) | Truncation test in CI; reviewers check `fit()` only sees train |
| QQQ's strong uptrend makes buy-and-hold hard to beat | Compare on risk-adjusted metrics (Sharpe, max drawdown, Calmar), not just return. Analyze 2022 separately |
| Integration surprises late | Contracts frozen on day 1; end-to-end checkpoint on Dec 6 |
| Member unavailable | Each WS has a named backup (the reviewer from §6) |
