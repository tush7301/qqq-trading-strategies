# Slot 2 (neural network): pre-registered protocol

Written **before** any neural-network result was computed, so the choices below cannot be bent to fit the results. Same rules as Slot 1 and Plan §3–4.

## Data and periods
- QQQ daily OHLCV and the 13-week T-bill series, both committed in `data/raw/`.
- **Train** runs 2005-01-03 → 2018-12-31. **Validation** runs 2019-01-02 → 2021-12-31.
- **Test (2022 →) is never loaded by any experiment in this folder.**
- Costs are 5 bps one-way. Cash earns T-bills, and Sharpe is in excess of T-bills (the shared engine).

## Model selection inside train (walk-forward)
Hyper-parameters are chosen **only on train**, using four expanding-window folds. Each model is fitted from scratch on the fit window, then trades the evaluation window:

| Fold | Fit window | Evaluation window |
|---|---|---|
| 1 | 2005–2010 | 2011–2012 |
| 2 | 2005–2012 | 2013–2014 |
| 3 | 2005–2014 | 2015–2016 |
| 4 | 2005–2016 | 2017–2018 |

- **Selection metric:** the Sharpe ratio of the four evaluation windows joined end to end (2011–2018, all out of sample with respect to each fit), after costs.
- **Early stopping** uses the **last 15% of each fit window**, in time order and never shuffled.
- **Seeds:** every configuration is trained with 5 seeds. The strategy trades the **average output of the 5 seeds** (a seed ensemble), and the spread across seeds is reported.

## Leakage controls
- Every feature at date *t* uses data up to the close of *t* only. This is unit-tested by truncation.
- Targets (next-day return or direction) are built **inside `fit()` from the training frame only**. The last row of each fit window has no target and is dropped (it is never labelled 0).
- Scalers and standardisation statistics come from the fit window only.
- The engine applies the one-day lag; the strategy never shifts positions itself.
- `scripts/leakage_audit.py` (PR #17) must pass. It runs truncation, alternate-future, hidden-data-access and random-walk placebo checks.

## Candidate families
| | Family | Inputs | Output → position | Loss |
|---|---|---|---|---|
| **A** | Direction classifier (the brief's design) | `lagged_ohlc_features` (5/10/20 lags) | P(up) → long if P > threshold | binary cross-entropy |
| **B** | Same, LSTM (PyTorch, research only) | 20–60-day sequences of the daily features | P(up) → long if P > threshold | binary cross-entropy |
| **C** | Sharpe-optimised network (Deep-Momentum-Network style; Lim, Zohren & Roberts 2019) | multi-horizon volatility-normalised returns, MACD trend signals, volatility | network output **is** the position, in [0, 1] | −Sharpe + turnover penalty |
| **D** | Neural volatility forecast used for sizing | HAR-style realised/range volatility lags | forecast σ̂ → min(1, σ_target / σ̂), always long | MSE on log volatility |

Grids are small on purpose (fewer than 20 configurations per family) to limit selection bias.

## Decision ladder for the slot (Slot 1's rule)
1. **v1 = family A** with its best train configuration. This is the brief's design and is always reported.
2. **v2 = the best of C and D** on the train walk-forward. It **replaces** v1 as the default **only if** both hold:
   - its train walk-forward Sharpe beats v1's, and
   - on validation (looked at once) its Sharpe is no worse than v1's − 0.05 **and** its max drawdown is no worse than v1's.
3. **v3 = an LSTM version (B, or C with an LSTM)** is adopted over v2 only under the same two conditions. It must also run without PyTorch in CI, or the PyTorch dependency must be approved by the team.
4. Whatever wins is reported **next to** buy-and-hold and every rejected variant. "The NN found no edge" is a valid outcome and will be reported as such.

---

## Addendum (written after the first validation look, so it is post-hoc)
**Finding:** the family-C network's Sharpe loss doesn't change if every position is multiplied by the same constant, so its *level* of exposure is arbitrary. It ends up wherever initialisation and weight decay leave it (≈30% on train, ≈22% on validation), which gives a low CAGR. Lim et al. (2019) handle this by rescaling positions to a volatility target.

**Variant v2b** applies the same rescaling:
- `scale="vol"`: inside `fit()`, compute k = (QQQ's train-period volatility) ÷ (volatility of the network's own train-period strategy).
- Trade min(1, k × output).
- Nothing else changes.

**Rule, fixed before running v2b:** v2b replaces v2 as the default if its train walk-forward Sharpe is ≥ v2's − 0.05. The point is to make exposure meaningful, not to raise Sharpe. Its validation result is reported but **marked post-hoc**, because the family's validation numbers had already been seen.
