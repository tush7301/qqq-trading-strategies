# Strategy research: what systematic traders use, and what works on QQQ

**Questions:** Which strategies do algorithmic traders actually use? Which are most widely used? Which are likely to give us the best results on QQQ?

**Method:** a literature review, then a screen of the textbook version of each strategy through our own engine. The screen uses no tuning, 5 bps costs, cash earning T-bills and Sharpe in excess of T-bills. It covers **train and validation only** and never touches the test period.

Reproduce: `python scripts/screen_strategies.py` (≈15 s). It writes `results/research/strategy_screen.md`.

---

## TL;DR

1. **Volatility management is the one consistent improvement over buy-and-hold for an equity index.** It means holding less QQQ when recent volatility is high, and it is supported both by the literature and by our data. It doesn't earn *more*. It earns about the same with roughly half the drawdown, so the Sharpe ratio rises. Slot 1 v3 already includes it, and that is most of why v3 beats buy-and-hold (see §3.2).
2. **Trend following** is the most widely used systematic style. On a single stock index its main benefit is avoiding long bear markets (2008), not a higher Sharpe.
3. **Predicting next-day direction with ML or a neural net is a coin flip on QQQ**, with a validation AUC of 0.50. That matches the literature. A more promising neural-net design is forecasting **volatility** and using it to size the position (§4).
4. Short-term mean reversion (RSI(2)), turn-of-the-month and overnight-only trading are popular, but they are weak or impractical for us.
5. None of the Sharpe differences are statistically significant: the bootstrap confidence intervals are about ±0.5 on train and ±1.1 on validation. The robust gains are smaller drawdowns and lower volatility.

---

## 1. What "algorithmic trading" mostly is

By volume, most algorithmic trading is not about predicting prices:

- **Market making and high-frequency trading** account for roughly half of US equity volume since 2010. They need sub-second data and exchange co-location.
- **Execution algorithms** (VWAP, TWAP, implementation shortfall) split large institutional orders to reduce market impact. They decide *how* to trade, not *whether* to.
- **Statistical arbitrage and pairs trading** need many related assets.

None of these apply to one ETF on daily bars. The families that do apply are below.

## 2. Strategy families that fit our setup (one ETF, daily bars, positions in [−1, 1])

| Family | Idea | What the literature says | Caveats | Our slot |
|---|---|---|---|---|
| **Trend following / time-series momentum** | Hold when the recent trend (moving average or trailing return) is up, otherwise go to cash | The most widely used systematic style; it is what CTAs and managed-futures funds run. Hurst, Ooi & Pedersen find it profitable in every decade across 67 markets back to 1880, and strongest in bear markets. Faber: below the 10-month SMA, US stock returns were lower and volatility higher | For timing a *single* stock index, Zakamulin finds no statistically significant outperformance in the second half of his sample, and argues published results are inflated by data mining and by ignoring frictions. It lags at turning points and gets whipsawed in sideways markets | 1 (MA), 4 (momentum) |
| **Volatility targeting** | Position = min(1, target vol ÷ recent realized vol) | Moreira & Muir: Sharpe ratios rise 50–100% for the market and factor portfolios. Harvey et al. (Man AHL): for US equities, Sharpe rises from 0.40 to ≈0.50 and the left tail gets thinner. It works for "risk assets" (equities, credit) because volatility rises when prices fall (the "leverage effect"); there is no gain for bonds, FX or commodities | Cederburg et al.: out of sample, volatility-managed *factor* portfolios beat the unmanaged ones in only 53 of 103 cases (8 significant). The equity-market case is the strongest one. Without leverage it can only *reduce* exposure, so it lags in calm bull markets | 4, and Slot 1 v3 already uses it |
| **Short-term mean reversion** | Buy after sharp short-term drops (RSI(2) < 10, %B < 0); sell on recovery | Popularised by Connors & Alvarez (2008), but their evidence is 1995–2007 backtests without transaction costs or data-mining corrections | Low time in the market. Gains are small and sensitive to costs | 3 |
| **Calendar effects** | Turn of the month: hold from the last trading day through the 3rd trading day | McConnell & Xu: from 1926 to 2005, essentially all of the US equity premium was earned in that 4-day window | Weak on QQQ in 2005–18 (§3). Easily data-mined | — |
| **Overnight vs intraday** | Hold only from close to open | Lou, Polk & Skouras: roughly 60% of the US equity premium is earned overnight | Needs two trades a day, ≈25%/yr in costs at 5 bps each way, and our engine is close-to-close | — |
| **ML / neural-net direction prediction** | Classify whether the next day is up or down from lagged prices or indicators | Welch & Goyal: none of the classic return predictors beat the historical mean out of sample. Gu, Kelly & Xiu: even the best ML models reach only ≈0.3–0.4% monthly out-of-sample R² at the stock level. The famous LSTM result (Fischer & Krauss, Sharpe 5.8 before costs) is cross-sectional across 500 stocks, not timing one index | Predictive edge on one index at a daily horizon is tiny. Overfitting is the main risk | 2 (NN), 5 (ML) |

**Overfitting applies to everything above.** McLean & Pontiff find published predictors earn about 26% less after the paper's sample period ends and 58% less after publication. Bailey, Borwein, López de Prado & Zhu show that with enough tried variants the best backtest is likely to underperform out of sample. That is why our plan fixes rules before looking at results and opens the test set once.

---

## 3. The screen on QQQ

Textbook settings, no tuning: 5 bps one-way costs, uninvested cash earns 13-week T-bills, and Sharpe is in excess of T-bills. Sharpe CIs are 95% moving-block bootstrap intervals.

### 3.1 Leaderboard

**Train (2005–2018)**

| Strategy | CAGR | Ann. vol | Sharpe | Sharpe 95% CI | Max DD | Calmar | Exposure | # trades |
|---|---|---|---|---|---|---|---|---|
| Slot 1 MA v3 (team default) | 12.1% | 14.8% | **0.76** | [0.31, 1.28] | −26.8% | 0.45 | 82% | 114 |
| Vol target 15%, always long | 11.0% | 14.1% | 0.73 | [0.25, 1.26] | −30.4% | 0.36 | 85% | 1813 |
| Slot 1 MA v2 (no vol cap) | **12.6%** | 16.5% | 0.73 | [0.29, 1.22] | −30.6% | 0.41 | 86% | 38 |
| Trend blend 1/3/12m × vol target | 7.6% | 10.3% | 0.65 | [0.16, 1.15] | −17.5% | 0.43 | 65% | 1793 |
| SMA200 × vol target | 8.1% | 11.9% | 0.61 | [0.12, 1.13] | −20.0% | 0.40 | 69% | 1212 |
| Trend blend 1/3/12m | 7.8% | 11.9% | 0.59 | [0.11, 1.11] | −28.0% | 0.28 | 72% | 489 |
| **Buy & hold** | 11.1% | 20.2% | 0.56 | [0.11, 1.10] | −53.4% | 0.21 | 100% | 1 |
| SMA200 filter (Faber) | 8.3% | 13.8% | 0.56 | [0.04, 1.08] | −26.3% | 0.31 | 76% | 94 |
| 12-month momentum | 8.5% | 15.5% | 0.53 | [0.09, 1.05] | −32.0% | 0.27 | 81% | 56 |
| RSI(2) + SMA200 filter | 5.0% | 7.6% | 0.52 | [0.12, 1.01] | −11.7% | 0.43 | 14% | 228 |
| RSI(2), no filter | 6.7% | 12.6% | 0.48 | [0.12, 0.86] | −19.5% | 0.34 | 22% | 310 |
| Turn of month | 2.0% | 8.5% | 0.13 | [−0.33, 0.66] | −26.1% | 0.08 | 19% | 337 |

**Validation (2019–2021)**

| Strategy | CAGR | Ann. vol | Sharpe | Sharpe 95% CI | Max DD | Calmar | Exposure | # trades |
|---|---|---|---|---|---|---|---|---|
| Vol target 15%, always long | 24.3% | 14.9% | **1.48** | [0.40, 2.59] | −13.7% | **1.78** | 78% | 489 |
| Slot 1 MA v3 (team default) | 28.1% | 17.8% | 1.44 | [0.32, 2.55] | −19.0% | 1.48 | 88% | 45 |
| **Buy & hold** | **37.9%** | 24.9% | 1.38 | [0.29, 2.60] | −28.6% | 1.33 | 100% | 0 |
| Slot 1 MA v2 (no vol cap) | 35.6% | 24.7% | 1.33 | [0.28, 2.57] | −28.6% | 1.25 | 99% | 1 |
| SMA200 filter (Faber) | 27.4% | 19.9% | 1.28 | [0.12, 2.45] | −22.0% | 1.25 | 93% | 11 |
| SMA200 × vol target | 19.7% | 14.3% | 1.27 | [0.20, 2.37] | −12.4% | 1.58 | 75% | 444 |
| Trend blend 1/3/12m × vol target | 16.9% | 12.5% | 1.25 | [0.19, 2.42] | **−10.4%** | 1.63 | 69% | 489 |
| Trend blend 1/3/12m | 22.4% | 17.0% | 1.23 | [0.12, 2.46] | −18.9% | 1.18 | 85% | 84 |
| 12-month momentum | 25.4% | 23.8% | 1.04 | [−0.01, 2.46] | −31.4% | 0.81 | 97% | 15 |
| Turn of month | 8.5% | 11.0% | 0.72 | [−0.34, 1.78] | −7.9% | 1.07 | 19% | 72 |
| RSI(2) + SMA200 filter | 5.1% | 9.9% | 0.47 | [−0.26, 1.40] | −8.6% | 0.59 | 13% | 48 |
| RSI(2), no filter | 6.5% | 16.1% | 0.42 | [−0.06, 1.44] | −17.1% | 0.38 | 15% | 48 |

### 3.2 What drives Slot 1's edge?

| | Train Sharpe | Validation Sharpe | Validation max DD |
|---|---|---|---|
| Buy & hold | 0.56 | 1.38 | −28.6% |
| MA signal only (v2) | 0.73 | 1.33 | −28.6% |
| Vol cap only (always long) | 0.73 | **1.48** | **−13.7%** |
| MA signal + vol cap (v3) | **0.76** | 1.44 | −19.0% |

- On train, both ingredients help about equally: the MA signal by sitting out 2008, the vol cap by cutting exposure in turbulent periods.
- On validation the MA signal *alone* trails buy-and-hold, and the vol cap does the work.

**For the report:** say explicitly that most of v3's advantage comes from volatility management.

**For selection:** under Plan §4, a plain vol-targeting strategy in Slot 4 would tie with v3 on validation Sharpe (within 0.1) and win the tie-break on Calmar and simplicity.

### 3.3 Volatility targeting is robust, not a lucky parameter

Train Sharpe by target volatility and estimation window. Buy & hold is 0.56.

| | 10-day window | 20-day window | 60-day window |
|---|---|---|---|
| target 10% | 0.72 | 0.71 | 0.70 |
| target 15% | 0.71 | 0.73 | 0.72 |
| target 20% | 0.70 | 0.71 | 0.70 |
| target 25% | 0.67 | 0.69 | 0.66 |

- **Costs:** Sharpe goes 0.74 → 0.69 on train and 1.50 → 1.44 on validation from 0 to 20 bps. Turnover is 3.3× a year.
- **Rebalance band:** only rebalancing on moves of more than 10 points halves turnover to 1.6× a year, with the same Sharpe.

### 3.4 Can lagged OHLC predict tomorrow's direction?

Inputs are `lagged_ohlc_features(lags=10)`. Models are fit on train with a scaler fit on train, and scored on validation.

| Model | Validation AUC | Share of days predicted "up" |
|---|---|---|
| Logistic regression (L2) | 0.483 | 84% |
| MLP (32, 16), 5 seeds | 0.503 ± 0.014 | 57% |
| Base rate | 0.500 | 60% (actual up days) |

**No signal.** The logistic model mostly predicts "up", which makes it a noisy copy of buy-and-hold.

### 3.5 Where QQQ's return was earned

| | Overnight (close→open) | Intraday (open→close) | Close→close |
|---|---|---|---|
| Train | 325% | 2% | 338% |
| Validation | 73% | 55% | 162% |

This matches the overnight-premium literature. We can't trade it, for the cost and engine reasons in §2.

---

## 4. Recommendations

1. **Slot 1:** keep v3. In the report, attribute its gain mainly to the volatility cap (§3.2).
2. **Slot 4 (momentum / vol targeting):** it overlaps with Slot 1 v3, so make the difference explicit. Either:
   - (a) **pure volatility targeting, always long.** This is the cleanest test of Moreira & Muir / Harvey et al., and it doubles as an attribution benchmark for Slot 1. Or:
   - (b) **trend blend (1/3/12-month) × vol target.** This has the lowest drawdowns of any strategy invested most of the time (−17.5% train, −10.4% validation), at a lower Sharpe.

   Decide which before tuning.
3. **Slot 2 (neural net):** the brief requires a neural net, but direction prediction gives AUC ≈ 0.50 (§3.4).
   - Suggested design: **an NN that forecasts next-week realized volatility**, used to size the position in place of the 20-day estimate. Volatility is persistent and predictable, unlike direction.
   - Keep the direction classifier as the baseline and report its AUC honestly. "No predictable signal" is a valid finding.
4. **Slot 3 (mean reversion):** expect a low Sharpe and ~13% time in the market. Report it as a contrasting low-exposure style. The T-bill engine now credits its idle cash.
5. **Slot 5 (ML classifier):** same caution as Slot 2. A regime model (e.g. a 2-state HMM on returns and volatility) is closer to what works (being out of the market in high-volatility regimes) than next-day classification.
6. **Don't select on test-period knowledge.** We all know 2022 was a bear market. Choosing or tweaking strategies *because* they would have handled 2022 leaks the test set into selection.

---

## References

- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4). https://papers.ssrn.com/abstract=2326253
- Cederburg, S., O'Doherty, M. S., Wang, F., & Yan, X. (2020). On the performance of volatility-managed portfolios. *Journal of Financial Economics*, 138(1), 95–117. https://www.lehigh.edu/~xuy219/research/COWY.pdf
- Connors, L., & Alvarez, C. (2008). *Short Term Trading Strategies That Work*. TradingMarkets.
- Faber, M. T. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management*, 9(4). https://papers.ssrn.com/abstract=962461
- Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654–669. https://iwf.rw.fau.de/files/2015/12/11-2017.pdf
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2274. https://www.nber.org/papers/w25398
- Harvey, C. R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M., & Van Hemert, O. (2018). The impact of volatility targeting. *Journal of Portfolio Management*, 45(1), 14–33. https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf
- Hurst, B., Ooi, Y. H., & Pedersen, L. H. (2017). A century of evidence on trend-following investing. *Journal of Portfolio Management*, 44(1). https://fairmodel.econ.yale.edu/ec439/hurst.pdf
- Lou, D., Polk, C., & Skouras, S. (2019). A tug of war: Overnight versus intraday expected returns. *Journal of Financial Economics*, 134(1), 192–213. https://personal.lse.ac.uk/polk/research/TugOfWar.pdf
- McConnell, J. J., & Xu, W. (2008). Equity returns at the turn of the month. *Financial Analysts Journal*, 64(2), 49–64.
- McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*, 71(1), 5–32. https://ivey.uwo.ca/media/3775549/pontiff.pdf
- Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611–1644. https://www.nber.org/papers/w22208
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228–250.
- Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455–1508. https://www.nber.org/papers/w10483
- Zakamulin, V. (2015). A comprehensive look at the empirical performance of moving average trading strategies. SSRN working paper. https://papers.ssrn.com/abstract=2677212
