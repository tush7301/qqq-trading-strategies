# Slot 2 (neural network): complete review log

**Purpose.** This is a record of **everything** done for Slot 2 and the related research PRs, in order, including bugs, reruns, deviations from the protocol and decisions made after seeing results. A reviewer should be able to explain the work from this file alone, without guessing.

**Who.** Akshat Bhandari (Slot 2 owner, GitHub `akshatbhandari15`) directed and reviewed the work. It was carried out with Claude, an AI assistant, working in the team repo between **19 Nov 2026, 20:08 ET** and **20 Nov 2026, early morning ET**.

**Where.**
- **PR #17** (`research-strategy-screen`): strategy research and leakage audit.
- **PR #18** (`s2-nn`): Slot 2. This file.

Neither PR was merged at the time of writing. Every merge to `main` needs Tushar's approval (CODEOWNERS).

---

## 0. One-paragraph summary (safe to say out loud)
We implemented the brief's neural network (predict tomorrow's direction from recent OHLC data) and found it **does not work on QQQ**: validation AUC is 0.504, a coin flip. Following the literature (Lim, Zohren & Roberts 2019), we instead trained the network **on the Sharpe ratio**, so that it outputs the position directly. Without being told, it learned to hold less QQQ when volatility is high and slightly less after losses. A Sharpe-trained network's exposure *level* is arbitrary, because Sharpe doesn't change when every position is multiplied by the same constant. So we rescaled its output to QQQ's own volatility; **this step was decided after seeing validation results and is labelled post-hoc everywhere**. The final version (v2b):
- has a validation Sharpe of **1.53**, vs **1.38** for buy-and-hold;
- has a max drawdown of **−16.2%**, vs **−28.6%**;
- earns a lower return, **25.1%** vs **37.9%** a year;
- only **ties** buy-and-hold on the out-of-fold training period (0.89 vs 0.89).

None of these differences is statistically significant. The test period (2022 onwards) was never loaded.

---

## 1. Numbers you may quote, and the ones you must not
| Quote this | Value | Where it comes from |
|---|---|---|
| v2b validation Sharpe (excess of T-bills, after 5 bps) | **1.53** (95% CI 0.39–2.75) | `results/strategies/neural_net/v2b_scaled/metrics.md`, `research/v2b_scaled_posthoc.csv` |
| Buy-and-hold validation Sharpe | **1.38** (CI 0.29–2.68) | same |
| v2b validation CAGR / vol / max DD | 25.1% / 14.8% / −16.2% | same |
| Buy-and-hold validation CAGR / vol / max DD | 37.9% / 24.9% / −28.6% | same |
| **Walk-forward Sharpe 2011–2018** (honest train-period estimate): v2b / buy-and-hold | **0.89 / 0.89** (0.885 / 0.887) | `research/v2b_scaled_posthoc.csv`, `research/log.txt` |
| v1 (the brief's classifier) validation AUC | **0.504** | `research/validation_once.csv` |
| v1 validation Sharpe / max DD | 1.09 / −33.2% | `v1_direction/metrics.md` |
| COVID crash 20 Feb–23 Mar 2020: v2b / buy-and-hold | −15.8% / −27.9% | `research/stress_episodes.csv` |
| What v2b learned (train period): correlation of position with 60-day vol / with 3-month return | −0.79 / +0.40 | `make_figures.py` → `what_it_learned()` |

**Do not quote these as performance:**
- **"Train Sharpe" of a fitted network** (v1 1.64, v2b 0.72). The network was fitted on those same years, so these are **in-sample**. v1's 1.64 and its "+36% through 2008" are memorisation. Use the walk-forward numbers instead.
- **"The neural network predicts the market."** It does not. v2b is a learned *risk-sizing* rule.
- **v2b's validation result as a clean out-of-sample test.** The rescaling idea came after the first validation look (§6).
- **Numbers from the first, chat-only strategy screen** (§3.2). They used the old 0%-cash engine and were superseded.

**Definitions** (used everywhere unless stated otherwise):
- **Sharpe:** annualised (√252) mean ÷ std of daily returns **in excess of the 13-week T-bill**, net of **5 bps** one-way costs. Uninvested cash earns the T-bill rate (shared engine, PR #13).
- **Walk-forward Sharpe:** four models fitted on 2005–2010, 2005–2012, 2005–2014 and 2005–2016, each trading only the next two years. The four out-of-fold windows (2011–2018) are joined and scored as one series.
- **Validation:** 2019-01-02 to 2021-12-31. A model fitted on all of train (2005–2018) trades it.
- **CI:** 95% moving-block bootstrap of the Sharpe ratio, 2,000 resamples, 20-day blocks (`qqq.metrics.bootstrap_sharpe_ci`, the repo default).
- **Max DD:** largest peak-to-trough fall of the equity curve within the period.

---

## 2. Data, environment, constraints
- **Data:** only the committed `data/raw/QQQ.csv` (Yahoo Finance via `yfinance`, 2005-01-03 to 2026-10-30) and `data/raw/IRX.csv` (13-week T-bill yield), both added by Tushar.
- **Firecrawl was requested to fetch data, but could not be used.**
  - The CLI was installed (`firecrawl-cli` 1.30.0), but the workspace's network policy blocks `api.firecrawl.dev` (HTTP 403 from the egress proxy).
  - Stooq, Yahoo, FRED, CBOE and Nasdaq were all unreachable too.
  - **The data was therefore not cross-checked against a second source.**
- **Software:** Python 3.13.18, NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, SciPy 1.18.1, matplotlib 3.11.2.
- **PyTorch 2.15.0** (the CUDA build from PyPI, run on CPU, because the CPU-only index was blocked) was used **only** for the research LSTMs. Slot 2 itself is pure NumPy, so CI needs nothing new.
- **Machine:** 2 CPUs, no GPU. Every run is seeded. A from-scratch rerun reproduced every saved table exactly, three separate times (§9). Results on other hardware or library versions could differ in the last decimals.
- **Test period:** no script in this work loads data after 2021-12-31. `nn_research.py` loads with `end=2021-12-31` and asserts it. Exported `positions.csv` files end on 2021-12-31.

---

## 3. Chronology

### 3.1 Repo review and issue proposals (19 Nov, ~20:08; no code)
- Read the project brief, the plan, all code, and issues/PRs #1–#12. Ran the tests (48 passed, 8 expected failures for the empty slots).
- **Proposed, but did not file**, new issues:
  - kickoff with owners, dates and milestones
  - cash earning T-bills
  - deciding how fitted models are refit before the test run
  - building every final-run output before the final run
  - drafting report §1–3 early
  - a random-walk placebo test
- Also suggested improvements to #3, #4, #6–#10.
- Flagged that **"Sunday, December 22nd" in the brief is a Tuesday in 2026** (it was a Sunday in 2024). The deadline should be confirmed with the professor.
- Tushar implemented T-bill cash in PR #13 (merged 19 Nov, ~20:17 ET). **None of the proposals were filed as issues.**

### 3.2 Strategy research, first screen (19 Nov, ~20:16; chat only)
- Literature search on algorithmic-trading strategies (sources in `docs/STRATEGY_RESEARCH.md`).
- Ran textbook strategies on train and validation **with the old engine** (cash earned 0%; Sharpe on raw returns). For example, buy-and-hold's train Sharpe was 0.62 there, and the always-long vol target was 0.80 / 1.53.
- **These numbers appeared only in chat and are superseded** by §3.3.
- This first screen also fitted a simple sklearn MLP direction classifier on validation. Its AUC was 0.503 (logistic regression: 0.483). **This validation result was seen before the Slot 2 protocol existed** (see §5).

### 3.3 PR #17: research doc and screen (19 Nov, ~20:20–20:26)
- `main` had meanwhile gained the T-bill engine (#13) and Slot 1 v3 (#14). The screen was rewritten as `scripts/screen_strategies.py` and **rerun on the new engine**.
- Wrote `docs/STRATEGY_RESEARCH.md`, added a README link, and opened PR #17. CI passed.

### 3.4 Firecrawl attempt, leakage audit, issue #4 (19 Nov, ~20:28–20:35)
- Firecrawl attempt failed (§2).
- `main` gained #15 and #16. PR #17 was rebased; the README change merged cleanly, and the branch was force-pushed.
- **`scripts/leakage_audit.py`** (in PR #17) runs four checks:
  1. truncation
  2. alternate future
  3. hidden data access
  4. random-walk placebo

  Two deliberately leaky "control" strategies test the checks themselves.
- **Bug found and fixed:** in the first version, check 2 missed the control that peeks at tomorrow's return. Both random alternate futures happened to start with an up day, matching the real data at all three cut dates. A **mirrored** future (real returns with the sign flipped) was added; the control is now caught by checks 1, 2 and 4.
- **Real leak found and fixed:** the screen's turn-of-month rule used the future rows of the price index to decide "is tomorrow the last trading day?". It now uses a Mon–Fri calendar known in advance. Its numbers changed (train Sharpe 0.13 → 0.18; validation 0.72 → 0.58).
- At Tushar's request, **issue #4 was assigned to `akshatbhandari15`**, with a comment claiming the slot.

### 3.5 Slot 2 (19 Nov, ~20:40–21:07)
In this order:
1. **Literature search on neural networks for trading.** Notes and sources are in `docs/NEURAL_NETWORK_RESEARCH.md`. Key sources:
   - Lim, Zohren & Roberts (2019): Sharpe-trained Deep Momentum Networks. Details were read from the arXiv PDF: inputs, Sharpe loss, turnover regularisation, and that results are on 88 futures and break even at about 2–3 bps.
   - Bucci (2020) and Christensen et al. (2023): NN volatility forecasting.
   - Wood et al. (2021, 2022); Kelly, Malamud & Zhou (2024); Zeng et al. (2023); Sezer et al. (2020); a 2025 reproducibility audit.
2. **Created branch `s2-nn` from `main`** (commit `bc34c49`).
3. **Wrote `PROTOCOL.md` before any neural-network result was computed.**
   - ⚠️ It was committed in the **same commit** as the results (`04ee522`), so git history cannot prove the ordering; this log is the record.
   - Also note what had **already** been seen: the screen's validation AUC for a simple MLP classifier (§3.2), and the screen's validation results for volatility targeting.
4. **Implemented `src/qqq/strategies/neural_net.py`** (details in §4). Checked the gradients of both losses against finite differences: maximum relative error 2e-7 (cross-entropy) and 1e-7 (Sharpe).
5. **Smoke tests, inside train only, with untuned settings:**
   - One fit on 2005–2016, traded 2017–2018 (the evaluation window of walk-forward fold 4): direction mode Sharpe 1.37, Sharpe mode 0.65. These are only "does it run" checks.
   - A training-curve trace (fit 2005–2016, last 15% held out), **without** early stopping:
     - Direction: held-out loss bottomed at epoch ~30 and then rose, while training loss fell to 0.35 by epoch 600.
     - Sharpe-trained: in-sample Sharpe rose to 4.6, while held-out Sharpe was best at epoch 1.
   - ⚠️ The **grid values** (L2 of 1e-3 or 1e-1, turnover penalty of 5 or 25 bps, rebalance band of 0 or 10%) were chosen **after** seeing this trace, which is train-only information. The protocol only fixed "small grids".
   - A one-fold smoke test of family D and the LSTMs on 2017–2018: D-MLP 0.74, D-HAR 0.68, direction LSTM (63 days) always in cash, Sharpe LSTM (63 days) 0.81.
6. **First full run of `nn_research.py --lstm`.** The walk-forward finished. It then **crashed in the validation stage** (an LSTM config was parsed with `hidden` as a tuple). At that point validation backtests for buy-and-hold and family A existed in memory, but **nothing was printed or saved**. Fixed.
7. **Second full run.** The walk-forward was identical to run 1. It crashed again **after all validation backtests were computed but before anything was printed** (a read-only NumPy array in the AUC step). Fixed. A `--validation-only` mode was added to reuse the saved train-stage results.
8. **`--validation-only`: the first and only look at validation for families A–D** (table in §6). Decision ladder applied: v1 = A, v2 = C (§6).
9. **Inspected what family C learned.** Average position on validation 22.7%, with 8 position changes. On the raw (unscaled, uncapped) output over 2007–2021, correlations were:
   - −0.71 with log volatility
   - +0.20 to +0.25 with 1-, 3- and 6-month normalised returns
   - +0.23 with the fastest MACD

   These different measurements also appear in chat. The report uses the v2b / train-period definition in §1.
10. **Identified that the Sharpe loss is scale-invariant. Wrote the post-hoc addendum** in `PROTOCOL.md`, *with the adoption rule*, **before** running v2b. Implemented `scale="vol"`, ran the walk-forward (0.885, rule satisfied), then validation (1.53).
11. **Set the class defaults to v2b.** Because defaults changed, every grid in `nn_research.py` was pinned to explicit values (`band`, `scale`), so the search reproduces regardless of defaults.
12. **Tests:** 22 new tests in `tests/test_neural_net.py`. The commit message and the first PR #18 description said 23; that was a miscount, and the PR description was corrected. One failed at first because a local `import qqq.data` shadowed the `qqq` fixture; renamed. Full suite: 91 passed, 6 expected failures (Slots 3–5).
13. **Third full run, from scratch, with pinned grids.** Every walk-forward, validation and v2b number was **identical** to runs 1–2 and to `--validation-only`.
14. **Exported** v1_direction, v2_sharpe and v2b_scaled with the team's `scripts/export_results.py` (train + validation only).
15. **Figures** (`make_figures.py`).
    - First drafts used a second y-axis for the price and had overlapping labels. Both were fixed (no dual axes).
    - The 2008 stress window was first 1 Oct 2007 – 9 Mar 2009 (buy-and-hold −49.7%). It was changed to **31 Oct 2007 – 9 Mar 2009** so the numbers match Slot 1's §4.1 table exactly (buy-and-hold −52.3%, MA v3 −25.7%).
16. **Claim check.**
    - "Held-out loss never beats a coin flip" was verified precisely: the best held-out cross-entropy is 0.6936, equal to a constant forecast of the training up-day share (0.6936) and above ln 2 (0.6931).
    - The docs' earlier "0.70" came from a different configuration and was corrected.
17. **Leakage audit on Slot 2** (PR #17's script run against this branch): all strategies pass. Slot 2's placebo edge is +0.03 Sharpe (t = 1.1).
18. **Wrote** the results README, the report section 4.2, `docs/NEURAL_NETWORK_RESEARCH.md`, `tuning.ipynb` (display only) and the README slot row. The `pyproject.toml` comment now says PyTorch is research-only.
19. **Checked** that no file contains test-period data, ran `run_all.py` (the whole bake-off, including Slot 2, took about 3 s), committed, pushed and **opened PR #18** (CI passed). Ticked the done items on issue #4 and appended a status line. "Merged" stays unticked.

### 3.6 Review pass (20 Nov, early morning; this file)
- **Found an inconsistency:** the research script's validation CIs used **1,000** bootstrap resamples, while the README, report and PR quoted **2,000**-resample CIs (the repo default). Standardised to 2,000 and regenerated. Only the CI column changed; e.g. buy-and-hold [0.29, 2.60] → [0.29, 2.68].
- The same fix was applied to PR #17's screen. CIs changed by at most 0.10, and an appendix records PR #17's history.
- `--validation-only` had overwritten `log.txt`. It now writes `log_validation_only.txt`.
- **Fourth full run from scratch:** every table identical again (only the CI column differs from runs 1–3, by design).
- The notebook's "about 1,000 networks" was corrected to **about 850** (§5).

---

## 4. Exactly what the code does (`src/qqq/strategies/neural_net.py`)

**Network**
- Fully connected, tanh hidden layers and a sigmoid output.
- Weights initialised N(0, 1/fan-in). Full-batch Adam (lr 3e-3, β = 0.9 / 0.999) with L2 weight decay.
- Early stopping on the **last 15%** of the fit window (time order), patience 60, maximum 600 epochs. The best held-out weights are kept.
- 5 seeds (42–46), outputs averaged.

**Inputs**
- **Direction mode:** `lagged_ohlc_features(lags)` from the shared library: log return, range, body, gap, 20-day volume z-score, each lagged 0 to lags−1.
- **Sharpe mode:** `momentum_features`:
  - returns over 1, 21, 63, 126 and 252 days, each divided by (60-day EWM daily vol × √h)
  - three MACDs: EMA(8)−EMA(24), EMA(16)−EMA(48), EMA(32)−EMA(96), with EMAs as pandas `ewm(span=S)`, each divided by the 63-day std of price, then by its own 252-day std. The papers define the EMA time-scale slightly differently, so these approximate Baz et al.'s signals rather than replicate them exactly
  - log annualised volatility
- Warm-up is 346 trading days: the first complete feature row and first position are on 2006-05-18. Before that the strategy holds T-bills, and those days are inside the train metrics.

**Preprocessing**
- z-score with the mean and std of the **whole fit window** (this includes the early-stopping holdout: a tiny leak into early stopping, none into evaluation), clipped to ±5.

**Targets** (built inside `fit()` from the training frame only)
- The next-day return. The last training day has none and is **dropped**.
- Direction mode: label = 1 if the return > 0. The threshold is 0.5, or "base" = the training up-day share.
- Sharpe mode: loss = −√252 · mean(R) / std(R), where R_t = position_t × return_{t+1} − c·|Δposition_t|. **c (5 or 25 bps) is a training-only turnover penalty.** The backtest charges the real 5 bps.

**Positions**
- Direction: 1 if the average P(up) > threshold, else 0.
- Sharpe: the average output in [0, 1]. With `scale="vol"`: min(1, k × output), where k = std(train returns) ÷ std(output × train returns), computed in-sample on the training rows. For the final model, **k = 3.84**.
- `band = 0.10`: the held position only changes when the new value moves by more than 0.10, or when crossing zero.
- **Long-only (0 to 1).** Lim et al. use −1 to 1. No leverage.

**Final defaults (v2b):** `mode="sharpe", hidden=(32, 16), turnover_bps=25, l2=0.1, band=0.10, scale="vol", n_seeds=5, seed=42`.

---

## 5. Protocol: what was planned vs what was actually done
| Planned (`PROTOCOL.md`) | Actually done | Deviation? |
|---|---|---|
| Hyper-parameters chosen on 4 walk-forward folds inside train | Yes | – |
| 5 seeds per configuration | Yes for families A, C, D; **3 seeds for the LSTMs** (compute) | ⚠️ minor |
| Small grids (< 20 per family) | A: 12, C: 16, D: 6, B: 2, C-LSTM: 2 | – |
| Grid values fixed in advance | The *size* was; the *values* were chosen after the train-only training-curve trace (§3.5 step 5) | ⚠️ disclosed |
| Validation looked at once | Once for families A–D (step 8), plus once for post-hoc v2b. Run 4 recomputed the same numbers (identical) | – |
| Nothing about validation known in advance | **Not true:** the screen (§3.2) had shown a simple MLP classifier's validation AUC ≈ 0.50 and volatility targeting's validation Sharpe | ⚠️ disclosed |
| Family D: NN volatility forecast | Also ran non-NN references (HAR; 20-day and EWMA volatility sizing, "D-baseline"). They were references only and **not eligible** for the slot | – |
| Family D training | sklearn `MLPRegressor` with manual time-ordered early stopping (patience 20, up to 300 passes), not the NumPy network | note |
| Ladder: v2 replaces v1 only if it is better on train *and* not worse on validation | C beat A on train **0.917 vs 0.908**, a very small margin. On validation, 1.51 vs 1.09 and −5.9% vs −33.2% drawdown → adopted | note: the train margin is within noise |
| Ladder: LSTM (v3) only if it beats v2 | C-LSTM 0.916 vs C 0.917 → **not adopted** (it would also need PyTorch in CI) | – |
| Post-hoc rescaling (addendum) | Rule written before the run; walk-forward 0.885 ≥ 0.917 − 0.05 → adopted | **post-hoc**, labelled everywhere |

**Why the rescaling is not a "Sharpe hack."** Sharpe doesn't change when every position is multiplied by the same constant, so rescaling *by itself* cannot raise Sharpe. It changes the exposure and return level (validation CAGR 8.6% → 25.1%). v2b's Sharpe differs from v2's (1.53 vs 1.51) only through the cap at 1 and the rebalance band. Its validation Sharpe was therefore largely predictable from v2's before it was run.

**Networks trained in total:** about 850 (≈ 240 in A, 320 in C, 160 in D, 54 LSTMs, 40 for seed dispersion, 40 for validation and v2b).

---

## 6. Results at each decision point
**Walk-forward 2011–2018 (train only); buy-and-hold = 0.887.** Best configuration per family:

| Family | Best configuration | Walk-forward Sharpe | Exposure | Position changes |
|---|---|---|---|---|
| A Direction MLP | lags 20, (32, 16), threshold 0.5, L2 0.01 | 0.908 | 87% | 348 |
| B Direction LSTM | 63-day sequences, 16 units | 0.181 | 39% | 91 |
| C Sharpe MLP | (32, 16), turnover 25 bps, L2 0.1, band 0.1 | **0.917** | 50% | 5 |
| C-LSTM | 63-day sequences, 16 units, turnover 25 bps | 0.916 | 53% | 2011 |
| D NN vol forecast | MLP (16), 5-day horizon | 0.850 | 97% | 72 |
| D-baseline (no NN) | 20-day realised vol | 0.911 | 96% | 36 |

- **Seed dispersion** (each seed alone, walk-forward): A has mean 0.66, sd 0.20 (0.46–0.97); C has mean 0.89, sd 0.05.
- **Volatility-forecast quality (QLIKE, lower is better, 21-day horizon, out of fold):** NN 0.282, HAR 0.305, EWMA 0.350, 20-day realised 0.468.

**Validation (looked at once):**

| | Validation Sharpe | 95% CI | CAGR | Max DD | Exposure |
|---|---|---|---|---|---|
| Buy & hold | 1.38 | [0.29, 2.68] | 37.9% | −28.6% | 100% |
| A (v1) | 1.09 | [−0.22, 2.68] | 23.9% | −33.2% | 79% |
| B | 0.87 | [−0.18, 2.08] | 11.1% | −13.2% | 61% |
| C (v2) | 1.51 | [0.39, 2.76] | 8.6% | −5.9% | 23% |
| C-LSTM | 1.45 | [0.30, 2.66] | 14.8% | −10.0% | 46% |
| D | 1.52 | [0.38, 2.62] | 31.6% | −18.9% | 93% |
| D-baseline | 1.47 | [0.33, 2.64] | 28.9% | −19.0% | 88% |
| **v2b (post-hoc)** | **1.53** | [0.39, 2.75] | 25.1% | −16.2% | 74% |

---

## 7. What was not done / open items
- **Test period:** not run. That happens once, at the team's final run.
- **Selection between slots:** not decided. Under Plan §4, v2b (1.53) and Slot 1 v3 (1.44) are within 0.1 on validation Sharpe. The tie-break is Calmar (v2b 1.55 vs MA 1.48), then the simpler model. That is for the team to decide.
- **Overlap:** Slot 1 v3 and v2b both end up cutting exposure in high volatility. Combining them is a refinement-phase idea, not done.
- **Not tried:** multi-asset training (where Deep Momentum Networks get their power), change-point features, long/short output, Transformers.
- **Issue proposals from §3.1:** not filed.
- **Data not independently verified** (Firecrawl blocked).
- **Both PRs need Tushar's review.**

---

## 8. Likely questions and correct answers
- **"Did the neural network predict the market?"** No. Direction prediction scored AUC 0.504. The final model is a learned sizing rule: it holds less when volatility is high.
- **"Why is the train Sharpe of v1 (1.64) so high?"** It is in-sample: the model was fitted on those years. Out of fold it is 0.91, and on validation 1.09, with a worse drawdown than buy-and-hold.
- **"Isn't the rescaling cheating?"** It was decided after the first validation look, so it is reported as post-hoc. It cannot raise Sharpe by itself, only the exposure level. Its adoption rule was written before it ran, and it used train data only.
- **"Did you tune on validation?"** No hyper-parameter was chosen on validation; selection used the train walk-forward. But validation *was* seen for related models before the protocol (§3.2), and once for the families (§6).
- **"Could there be look-ahead?"** We ran four independent leakage checks; all pass, including a random-walk placebo with an edge of +0.03. Targets and scalers come from training data only. Known small exceptions are listed in §4 (early-stopping holdout statistics).
- **"Why not an LSTM, since the paper says it's best?"** We tested it. On one asset it matched the MLP (0.916 vs 0.917) and needs PyTorch, so the protocol kept the MLP.
- **"How sure are you that v2b beats buy-and-hold?"** Not sure. The CIs overlap heavily, and out of fold on 2011–2018 it only ties buy-and-hold.
- **"What data did you use?"** The team's committed Yahoo Finance QQQ file and 13-week T-bill yields. Not re-downloaded.

---

## 9. Reproduce
```bash
pip install -e ".[dev]"                                   # + "torch" only for --lstm
python notebooks/02_neural_net/nn_research.py --lstm      # ~7 min; writes results/strategies/neural_net/research/
python notebooks/02_neural_net/make_figures.py            # ~10 s; figures + stress_episodes.csv
python scripts/export_results.py neural_net v1_direction --params '{"mode": "direction", "lags": 20, "hidden": [32, 16], "threshold": 0.5, "l2": 0.01, "band": 0.0, "scale": "none"}'
python scripts/export_results.py neural_net v2_sharpe --params '{"scale": "none"}'
python scripts/export_results.py neural_net v2b_scaled
pytest -q                                                 # 91 passed, 6 xfailed
python scripts/leakage_audit.py                           # from PR #17
```

## 10. File map
| File | What it is |
|---|---|
| `src/qqq/strategies/neural_net.py` | The strategy (NumPy MLP, both modes, v2b defaults) |
| `tests/test_neural_net.py` | 22 tests |
| `notebooks/02_neural_net/PROTOCOL.md` | Pre-registered rules + post-hoc addendum |
| `notebooks/02_neural_net/REVIEW_LOG.md` | This file |
| `notebooks/02_neural_net/nn_research.py` | Walk-forward search, validation, v2b |
| `notebooks/02_neural_net/nn_lstm.py` | LSTM variants (PyTorch, research only) |
| `notebooks/02_neural_net/make_figures.py` | Figures and stress table |
| `notebooks/02_neural_net/tuning.ipynb` | Displays saved results; trains nothing |
| `results/strategies/neural_net/` | Exported v1 / v2 / v2b, `research/` tables, README |
| `report/sections/04_2_neural_network.md` | Report section 4.2 |
| `docs/NEURAL_NETWORK_RESEARCH.md` | Literature review and findings |
