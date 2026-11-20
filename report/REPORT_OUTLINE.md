# Final Report Outline

Each author writes Markdown in `report/sections/`; the team representative assembles it.

| § | Section | Writer | Source material already in repo | Length |
|---|---|---|---|---|
| 1 | Trading problem & objectives: QQQ background, goal = risk-adjusted performance vs buy-and-hold | split at selection meeting | — | 1 p |
| 2 | Historical data: source, period, cleaning, per-period stats, train/val/test split | split at selection meeting | `data/README.md` | 1 p |
| 3 | Back-testing procedure: timing convention, look-ahead safeguards, costs, metric formulas | split at selection meeting | docstrings of `backtest.py`, `metrics.py` | 1 p |
| 4.1–4.5 | One section per strategy: intuition, method, parameter search, chosen params, train vs validation | slot owners | slot notebooks | 1–1.5 p each |
| 5 | Strategy selection: leaderboard, selection rule (Plan §4), why the winner won | representative | `results/metrics/leaderboard_validation.md` | 1 p |
| 6 | Out-of-sample results (test): full table for **all** strategies + benchmark (mean/std daily + annualized, CAGR, Sharpe, cum. return, max DD), equity curves, drawdowns, risk-return, cost sensitivity, Sharpe CI | split at selection meeting | `results/` after final run | 2–3 p |
| 7 | Conclusions: strengths, weaknesses and limitations of each strategy; next steps | everyone → representative edits | — | 1 p |
| A | Appendix: reproducibility steps, code map | representative | `README.md` | — |
