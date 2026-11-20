# Neural networks for trading QQQ: research notes

**Why this exists.** The brief names a neural-network strategy (Slot 2). This document reviews what the literature says about neural networks for trading, explains which designs we tested and why, and summarises what we found on QQQ.

**What we tested:**
- 5 model families, 40 configurations
- walk-forward selection inside train, validation looked at once
- test period never loaded

Code and the pre-registered protocol are in [`notebooks/02_neural_net/`](../notebooks/02_neural_net/). **A complete step-by-step record of what was done**, with deviations, bugs, reruns and post-hoc decisions, is in [`REVIEW_LOG.md`](../notebooks/02_neural_net/REVIEW_LOG.md). Numbers are in [`results/strategies/neural_net/`](../results/strategies/neural_net/README.md). The write-up is report §4.2.

---

## 1. The short version

| Question | Literature | Our QQQ evidence |
|---|---|---|
| Can a network predict tomorrow's direction of a broad index? | Mostly no. Classic predictors fail out of sample (Welch & Goyal 2008), and ML gains are small and come mainly from cross-sections of many stocks (Gu, Kelly & Xiu 2020) | **No.** Validation AUC is 0.504. Held-out loss never beats a coin flip, and the LSTM is worse |
| Is there a better way to use a network for trading? | Train it on the **trading objective** (the Sharpe ratio) and let it output the position: Deep Momentum Networks (Lim, Zohren & Roberts 2019) | **Yes, partly.** The Sharpe-trained network is stable across seeds and learned "hold less when volatile, a bit less after losses" |
| Can networks forecast volatility? | Yes. They match or beat HAR models, especially at longer horizons (Bucci 2020; Christensen, Siggaard & Veliyev 2023) | **Yes**, but better forecasts did **not** give better trading than 20-day realised volatility |
| Do bigger or sequence models (LSTM, Transformer) help? | On many assets they can (Lim et al. 2019; Wood et al. 2022). On generic time series, simple linear models often match Transformers (Zeng et al. 2023) | **No gain** on one asset: the Sharpe-trained LSTM scored 0.92 vs 0.92 for the MLP |
| What is the main risk? | Overfitting and leakage. Many published deep-learning trading results don't meet basic standards | In-sample Sharpe rose to 2.5 while held-out Sharpe fell below 0 |

**Final Slot 2 strategy (v2b):** a Sharpe-trained MLP whose output is rescaled to QQQ's volatility. Validation Sharpe 1.53 vs 1.38 for buy-and-hold, max drawdown −16% vs −29%. It only ties buy-and-hold out of fold on train, and the confidence intervals overlap.

---

## 2. A map of neural-network trading approaches

### 2.1 Predict, then trade (supervised forecasting)
The network predicts the next return (regression) or its sign (classification), and a fixed rule maps the prediction to a position. This is the brief's design and the most common one in the literature.

Reviews of the field (Sezer, Gudelek & Ozbayoglu 2020) catalogue hundreds of studies built on LSTMs, CNNs and deep belief networks. The well-known positive results are mostly **cross-sectional statistical arbitrage**:
- Fischer & Krauss (2018) report daily returns of 0.46% and a Sharpe of 5.8 *before costs*. Their strategy goes long and short across all S&P 500 stocks, ranking them against each other. That doesn't need the market's direction to be predictable.

For **aggregate market timing** the evidence is much weaker:
- Welch & Goyal (2008) find that none of the classic equity-premium predictors would have beaten the historical mean out of sample.
- Gu, Kelly & Xiu (2020) find the best ML models reach monthly out-of-sample R² of only about 0.3–0.4% for individual stocks.

**Why it struggles:**
- Daily index returns are close to unpredictable, and only about 55–60% of days are up.
- A classifier can reach "accuracy" by always predicting "up", which just copies buy-and-hold.
- Accuracy also ignores *how much* is won or lost on each day. Lim et al. make the same point: classification accuracy need not translate into trading profit.

### 2.2 Learn the position directly (end-to-end, Sharpe-optimised)
**Deep Momentum Networks** (Lim, Zohren & Roberts 2019):
- The network outputs the position in [−1, 1] and is trained by gradient descent on the **negative Sharpe ratio** of the resulting returns.
- Inputs are volatility-normalised returns over 1 day to 12 months and MACD trend signals (Baz et al. 2015).
- Returns are scaled to a 15% volatility target.

On 88 futures (1995–2015):
- The Sharpe-optimised LSTM reached a pre-cost Sharpe of 2.8, against about 1.2 for classic time-series momentum.
- Direct position output beat "predict then size" designs, and classification losses were worst.
- **Costs bite.** The LSTM's edge disappeared at about 2–3 bps per trade. The authors therefore add a turnover penalty to the loss, which keeps it profitable at 10 bps.

**Follow-ups from the same group:**
- *Slow Momentum with Fast Reversion* (Wood, Roberts & Zohren 2021) adds a change-point detection module so the model can switch between trend and reversal regimes.
- The *Momentum Transformer* (Wood, Giegerich, Roberts & Zohren 2022) uses an attention-LSTM hybrid that is also interpretable.

**Why this suits us:** the objective matches how we are judged (Sharpe after costs), and the turnover penalty handles our 5 bps costs.

**Why it is harder for us:** we have **one** asset, not 88. Less diversification means a noisier loss and fewer independent examples to learn from.

### 2.3 Use the network for risk, not direction
Volatility is persistent and therefore predictable, unlike direction:
- Bucci (2020) finds recurrent networks (LSTM, NARX) beat econometric models for forecasting realised volatility.
- Christensen et al. (2023) find off-the-shelf ML beats the HAR benchmark (Corsi 2009), especially at longer horizons.

A volatility forecast can drive position sizing: min(1, σ* / σ̂). That is the volatility management that improves equity Sharpe ratios (Moreira & Muir 2017; Harvey et al. 2018).

### 2.4 Bigger models
- Kelly, Malamud & Zhou (2024) argue that heavily over-parameterised models (random-feature networks with more parameters than observations) can *improve* market-timing Sharpe ratios out of sample. Their evidence uses monthly data and 15 macro predictors, not daily prices.
- In the other direction, Zeng et al. (2023) show that a one-layer linear model often matches Transformer forecasters on standard time-series benchmarks.

For a single daily price series, we expect architecture to matter far less than the objective, the inputs and regularisation.

### 2.5 Leakage and overfitting: the field's main failure mode
- Common leaks:
  - scalers fitted on the full sample
  - shuffled train/test splits
  - "next-day" labels computed on the full series and then sliced
  - early stopping on a random (not time-ordered) holdout
  - repeatedly peeking at the test set
- A 2025 compliance audit of forecasting papers against seven basic reproducibility standards found **none fully compliant** (QFRS framework, Brunel).
- With many configurations tried, the best backtest is likely to disappoint out of sample (Bailey et al. 2017).

---

## 3. What we built and why

| | Family | Rationale | Inputs | Output / loss |
|---|---|---|---|---|
| A | Direction MLP | The brief's design; the baseline | 5–20 days of returns, range, body, gap, volume z | P(up) / cross-entropy; long if P > 0.5 |
| B | Direction LSTM | "Is a sequence model the missing piece?" | 20 or 63-day sequences of daily features | same as A |
| C | Sharpe-trained MLP | Lim et al. (2019): optimise what we are judged on | volatility-normalised returns over 1d–12m, 3 MACDs, log volatility | position in [0, 1] / −Sharpe + turnover penalty |
| C-LSTM | Sharpe-trained LSTM | Lim et al.'s best architecture | same as C, 20 or 63-day sequences | same as C |
| D | NN volatility forecast → sizing | Volatility is predictable (Bucci 2020; Christensen et al. 2023) | HAR-style realised and range-based volatility lags, leverage-effect term | log vol over the next 5 or 21 days / MSE; position min(1, σ*/σ̂) |

Implementation choices that matter:
- **NumPy, not PyTorch, for the slot.** The MLP is about 100 lines with gradients checked against finite differences. Everyone can run it, and CI needs no 2 GB dependency. The LSTMs (research only) use PyTorch.
- **Seed ensembles.** Each configuration trains 5 seeds (3 for the research LSTMs) and trades their average output. Direction models vary a lot by seed (single-seed walk-forward Sharpe 0.46–0.97); Sharpe-trained ones don't (sd 0.05).
- **Time-ordered early stopping** on the last 15% of each training window.
- **Walk-forward model selection inside train.** Fits on 2005–10/12/14/16 each trade the next two years, giving a 2011–2018 out-of-fold score. Validation is used once, to confirm.

---

## 4. Findings on QQQ

1. **Direction prediction fails.**
   - The best held-out cross-entropy (0.6936) only matches a constant forecast of the training share of up days (0.6936). It never beats a coin flip (ln 2 = 0.6931), while training loss fell towards 0.1.
   - Validation AUC is 0.504.
   - The "best" direction model owes its walk-forward Sharpe (0.91 vs buy-and-hold 0.89) to being 87% invested.
   - Out of sample it lost −33% in the COVID crash, more than buy-and-hold.
2. **The Sharpe-trained network learns something sensible.**
   - Without being told, it holds less when volatility is high (correlation −0.79) and a little less after losses (+0.40 with the 3-month return).
   - That is volatility targeting plus trend: the two effects in Slot 1 v3 and in the literature.
3. **A subtle trap: Sharpe doesn't change when every position is multiplied by the same constant**, so the *level* of a Sharpe-trained position is arbitrary. Ours settled at about 25% invested and earned 8.6% a year on validation, despite a Sharpe of 1.51. Rescaling to a volatility target (as Lim et al. do) fixes this: v2b.
4. **Neural volatility forecasting improves forecasts but not trading.**
   - QLIKE at 21 days: 0.28 for the NN, 0.31 for HAR, 0.47 for 20-day realised volatility.
   - Walk-forward Sharpe: 0.85 for NN sizing vs 0.91 for 20-day sizing.
5. **LSTMs add nothing on one asset.** The Sharpe-trained LSTM scored 0.92 vs 0.92 for the MLP; the direction LSTM scored 0.18.
6. **Final v2b vs buy-and-hold:**
   - Validation Sharpe 1.53 vs 1.38, max drawdown −16% vs −29%, CAGR 25% vs 38%.
   - COVID crash: −15.8% vs −27.9%.
   - Out of fold on train it ties (0.89 vs 0.89).
   - Not statistically significant.
7. **The leakage audit passes.** Truncation, alternate future, hidden data access, and a random-walk placebo (edge +0.03, t = 1.1).

---

## 5. Ideas for the refinement phase and beyond (not done)
- **Multi-asset training.** Train the Sharpe network on several equity indices (or sector ETFs) and apply it to QQQ. This is where Deep Momentum Networks get their power. It needs new data, which must be committed and documented.
- **Change-point features** (Wood et al. 2021) to react faster after crashes. v2b's main weakness is slow re-entry (+7.6% vs +33.1% in the 2020 rebound).
- **Combine with Slot 1.** The network learned the volatility response and Slot 1 encodes the trend break. An ensemble of the two positions is the natural refinement-phase experiment (Plan §4).
- **Long/short output** ([−1, 1]). Not tested: QQQ's drift makes shorting costly, and the contract allows it.

---

## References
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4). https://papers.ssrn.com/abstract=2326253
- Baz, J., Granger, N., Harvey, C. R., Le Roux, N. & Rattray, S. (2015). Dissecting investment strategies in the cross section and time series. SSRN working paper.
- Bucci, A. (2020). Realized volatility forecasting with neural networks. *Journal of Financial Econometrics*, 18(3), 502–531. https://mpra.ub.uni-muenchen.de/95443/
- Christensen, K., Siggaard, M. & Veliyev, B. (2023). A machine learning approach to volatility forecasting. *Journal of Financial Econometrics*, 21(5), 1680–1727. https://arxiv.org/abs/2601.13014
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Fischer, T. & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654–669.
- Gu, S., Kelly, B. & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2274. https://www.nber.org/papers/w25398
- Harvey, C. R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M. & Van Hemert, O. (2018). The impact of volatility targeting. *Journal of Portfolio Management*, 45(1), 14–33.
- Hurst, B., Ooi, Y. H. & Pedersen, L. H. (2017). A century of evidence on trend-following investing. *Journal of Portfolio Management*, 44(1).
- Kelly, B., Malamud, S. & Zhou, K. (2024). The virtue of complexity in return prediction. *Journal of Finance*, 79(1), 459–503. https://www.nber.org/papers/w30217
- Lim, B., Zohren, S. & Roberts, S. (2019). Enhancing time-series momentum strategies using deep neural networks. *Journal of Financial Data Science*, 1(4). https://arxiv.org/abs/1904.04912
- Moreira, A. & Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611–1644.
- Sezer, O. B., Gudelek, M. U. & Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning: A systematic literature review 2005–2019. *Applied Soft Computing*, 90, 106181. https://arxiv.org/abs/1911.13288
- Welch, I. & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455–1508.
- Wood, K., Roberts, S. & Zohren, S. (2021). Slow momentum with fast reversion: A trading strategy using deep learning and changepoint detection. *Journal of Financial Data Science*. https://arxiv.org/abs/2105.13727
- Wood, K., Giegerich, S., Roberts, S. & Zohren, S. (2022). Trading with the Momentum Transformer: An intelligent and interpretable architecture. https://arxiv.org/abs/2112.08534
- Zeng, A., Chen, M., Zhang, L. & Xu, Q. (2023). Are Transformers effective for time series forecasting? *AAAI*, 37(9). https://arxiv.org/abs/2205.13504
- QFRS reproducibility standards for financial forecasting papers (Brunel University London, 2025). https://bura.brunel.ac.uk/bitstream/2438/33856/3/FullText.pdf
