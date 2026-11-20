# Implementation Plan: QQQ Trading Strategies

**Course:** Finance and Structuring for DS, Final Project
**Asset:** QQQ (Invesco QQQ Trust, Nasdaq-100)
**Weekly progress email:** sent by the team representative at the end of each week

---

## 0. Approach in one paragraph

All shared infrastructure is **already built and tested**: data, features, the backtest engine, metrics, the evaluation helpers, the comparison/leaderboard, and the charts. Each of the 5 team members now owns **one strategy slot** and builds it independently against the same interface. The pipeline runs every strategy under identical rules (same data, same costs, same timing) and ranks them on the **validation** period. At the **selection meeting** we pick the strategy to move ahead with, spend a week refining it, then open the sealed **test** period once to get the final numbers.

---

## 1. What the brief requires → where it is handled

| # | Requirement | Status / where |
|---|---|---|
| R1 | ≥ 2 strategies using different techniques | 5 slots: MA crossover, neural net, mean reversion, momentum/vol-target, ML classifier |
| R2 | Python implementation and backtesting | ✅ `qqq.backtest.run_backtest` |
| R3 | Expected return and std dev, **daily and annualized** | ✅ `qqq.metrics.compute_metrics` (mean, std, ×252, ×√252, CAGR) |
| R4 | Cumulative return, maximum drawdown | ✅ same |
| R5 | Comparison to buy-and-hold | ✅ `BuyAndHold` is always in the leaderboard |
| R6 | No look-ahead bias | ✅ the engine owns the 1-day lag; CI runs a truncation test on every strategy and feature |
| R7 | Separate training and testing periods | ✅ train / validation / test in `config.py`. The test period is excluded unless `--include-test` is passed |
| R8 | Report | Each slot owner writes their section; the representative assembles (§6) |
| R9 | Data and reproducible code | ✅ `data/raw/QQQ.csv` committed; `scripts/run_all.py` regenerates every table and figure |

---

## 2. What is already built

| Module | What it gives you |
|---|---|
| `qqq.data` | `load_prices()` loads the committed QQQ OHLCV (2005-01-03 → 2026-10-30, 5,492 days); `split()` returns train/val/test; a synthetic data option exists for noise tests |
| `qqq.features` | `sma, ema, rsi, macd, bollinger_pct_b, rolling_volatility, momentum, zscore, lagged_ohlc_features, technical_indicators, forward_return` (target only). All causal and tested |
| `qqq.backtest` | `run_backtest(prices, positions, cost_bps)` returns net/gross returns, equity, turnover; **uninvested cash earns the T-bill rate**. `result.metrics()` gives all metrics (Sharpe in excess of T-bills) plus # trades, holding period, exposure, turnover |
| `qqq.metrics` | `compute_metrics()` (12 metrics), `annual_returns`, `drawdown_series`, `bootstrap_sharpe_ci`, `excess_return_test` |
| `qqq.evaluate` | **The main tools for strategy owners:** `evaluate(strategy)`, `grid_search(cls, grid)`, `check_no_lookahead(strategy)`, `backtest_strategy()` |
| `qqq.reporting` | Leaderboard tables (CSV and Markdown), equity / drawdown / risk-return / annual-return charts |
| `scripts/run_all.py` | Runs every registered strategy and writes `results/`, skipping strategies that aren't implemented yet |
| `notebooks/00_quickstart.ipynb` | The develop → tune → validate workflow in 10 cells |

**Development-period benchmark (buy-and-hold, 5 bps costs):**

| Period | CAGR | Ann. vol | Sharpe | Max DD |
|---|---|---|---|---|
| Train 2005–2018 | 11.1% | 20.2% | 0.56 | −53.4% |
| Validation 2019–2021 | 37.9% | 24.9% | 1.39 | −28.6% |

Sharpe is in excess of the 13-week T-bill rate.

Validation was an exceptionally strong bull market. Beating buy-and-hold on raw return there will be hard, so we judge on **risk-adjusted** metrics and drawdown (see §4).

---

## 3. Strategy slots (one per team member)

Each slot is one file in `src/qqq/strategies/` plus one folder in `notebooks/`. The files already contain the class skeleton, default parameters, and a "suggested path" docstring. Owners are free to change the idea inside their slot, as long as it stays a *different technique* from the other slots.

| Slot | File | Technique | Why it's worth testing on QQQ |
|---|---|---|---|
| 1 | `moving_average.py` | SMA/EMA crossover (trend following). **Named in the brief** | Sidesteps long bear markets (2008, 2022), at the cost of lag and whipsaw |
| 2 | `neural_net.py` | MLP or LSTM on lagged OHLC features. **Named in the brief** | Tests whether nonlinear patterns in recent price action predict the next day |
| 3 | `mean_reversion.py` | RSI / Bollinger oversold entries, optional trend filter | Index ETFs show short-term reversal; few days in the market and a high hit rate |
| 4 | `momentum.py` | Time-series momentum and volatility targeting | Vol targeting cuts exposure in crashes, often improving Sharpe and drawdown |
| 5 | `ml_classifier.py` | Logistic regression / gradient boosting on indicators (or an HMM regime model) | An interpretable ML baseline to compare against the NN |

### Rules every slot follows
1. `generate_positions(prices)` returns values in **[−1, 1]**, where `position[t]` uses data up to the **close of t**. **Do not shift: the engine does.**
2. Anything learned from data (models, scalers, thresholds) is learned inside `fit(train_prices)`, using **train only**.
3. Tune on **train** with `grid_search(...)`, choose among the top candidates on **validation**, then set the chosen parameters as the class defaults.
4. Never evaluate on **test**: `grid_search` refuses it, and `evaluate` warns.
5. Use `config.RANDOM_SEED`, and keep results deterministic.
6. CI must stay green: once you implement your slot, the contract and look-ahead tests for it switch from xfail to pass automatically.

### Per-slot deliverables
| Phase | Deliverable |
|---|---|
| **2: Build** | v1 implemented, tests passing, merged to `main` (shows up in the first leaderboard) |
| **3: Tune** | Tuning done; final parameters set as defaults; a notebook with the parameter search and a validation table; results exported with `scripts/export_results.py <strategy> <label>` into `results/strategies/` |
| **4: Refine** | Report section (≈1–1.5 pages): idea and economic intuition, method and parameters, train-vs-validation results, strengths, weaknesses, limitations |

---

## 4. Choosing the strategy to move ahead with

The pipeline (`python scripts/run_all.py`) produces `results/metrics/leaderboard_validation.md` and the matching charts. At the **selection meeting** (end of Phase 3) the team picks the winner using this pre-agreed rule (agreed now, so we can't bend it after seeing results):

1. **Eligibility:** contract and look-ahead tests pass; results reproducible.
2. **Primary metric:** highest **validation Sharpe ratio** after 5 bps costs.
3. **Sanity checks** (a strategy failing two or more drops to the next candidate):
   - Validation max drawdown no worse than buy-and-hold's (−28.6%)
   - Not overfit: validation Sharpe ≥ 50% of train Sharpe, *or* the strategy beats buy-and-hold Sharpe in **both** periods
   - Robust to costs: validation Sharpe at 10 bps ≥ buy-and-hold's Sharpe (`cost_sensitivity_validation.md`)
   - At least 10 position changes in train (not a near-copy of buy-and-hold)
4. **Ties** (Sharpe within 0.1): prefer the higher Calmar, then the simpler model.

**After selection (Phase 4):** the whole team focuses on the winner. Options include sensitivity of the winner to its parameters, combining it with the runner-up (e.g. MA trend filter + vol targeting), walk-forward refitting, and bootstrap confidence intervals for its Sharpe (`bootstrap_sharpe_ci`). **The report still presents all implemented strategies**: the brief asks for a comparison, and a strategy that loses to buy-and-hold is a valid finding.

**Final run (start of Phase 5), once:**
```bash
python scripts/run_all.py --include-test
```
These test-period numbers are the headline results, whatever they turn out to be. No re-tuning afterwards.

---

## 5. Phases

| Phase | Milestone |
|---|---|
| **0: Infrastructure** | ✅ Repo, data, engine, metrics, evaluation, reporting, and CI done |
| **1: Kickoff** | Assign slots and the team representative. Everyone runs `00_quickstart.ipynb` |
| **2: Build** | Build v1 of each strategy. Ends with the **first leaderboard** |
| **3: Tune & select** | Tuning on train, selection on validation; params frozen. Ends with the **selection meeting → choose the winner** (§4) |
| **4: Refine** | Refine and stress-test the winner; robustness/significance; all report sections drafted |
| **5: Final run & report** | **Test set opened once.** Results tables and figures final; results and conclusions written; editing; fresh-clone reproducibility check; final review → **submit** |

The team sets the actual dates for each phase at kickoff, working back from the submission deadline and leaving some buffer at the end.

---

## 6. Report & team representative

The **team representative** (one of the five, in addition to their slot) sends the weekly emails using `docs/progress_reports/TEMPLATE.md`, archives each one in `docs/progress_reports/`, and assembles the report from `report/sections/`. Outline and page budget: `report/REPORT_OUTLINE.md`. Shared sections (intro, data, backtest procedure, results, conclusions) are split at the selection meeting. Most of the content for the data and procedure sections already exists in `data/README.md` and the docstrings of `backtest.py` and `metrics.py`.

## 7. Working agreements
- **Branch per slot** (`s1-ma`, `s2-nn`, `s3-meanrev`, `s4-momentum`, `s5-ml`); open a PR to `main` with 1 reviewer; CI must be green.
- **Stay in your files:** your strategy module and your notebook folder. Changes to shared infrastructure (`data, features, backtest, metrics, evaluate, reporting, config`) go through a PR that tags the whole team. New *features* are welcome in `features.py`, but add a causality test.
- **Notebooks** are for exploration; the final strategy logic lives in `src/`. Clear outputs before committing, except final figures.
- **Never commit test-period numbers before the final run.**

## 8. Risks
| Risk | Mitigation |
|---|---|
| Nothing beats buy-and-hold in the 2019–21 bull market | Expected for QQQ. Judge on Sharpe and drawdown; report honestly. The test period includes the 2022 bear market, where timing strategies can shine |
| Overfitting through extensive grid searches | Rank on train, choose on validation, sanity rules in §4, single test run |
| Hidden look-ahead (scaler fitted on all data, `forward_return` used as a feature, centered windows) | Truncation test in CI for every strategy; reviewers check that `fit()` only sees train |
| NN results vary by seed | Report mean ± sd over 5 seeds |
| A member drops out | Their slot is simply missing from the leaderboard. The brief only needs 2 strategies; slots 1 and 2 have priority |
