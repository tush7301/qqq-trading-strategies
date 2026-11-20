# Data

## `raw/QQQ.csv`
| | |
|---|---|
| Instrument | QQQ: Invesco QQQ Trust, Series 1 (tracks the Nasdaq-100) |
| Source | Yahoo Finance via `yfinance` (`auto_adjust=False`) |
| Downloaded | 2026-11-19 |
| Range | 2005-01-03 → 2026-10-30 (5,492 trading days) |
| Columns | `Date, Open, High, Low, Close, AdjClose, Volume` |
| Missing values | none |

- `Open/High/Low/Close` are **as traded** (split-adjusted only).
- `AdjClose` is also **dividend-adjusted**. All returns in the project are computed from `AdjClose`, so they are total returns.
- `features.lagged_ohlc_features` rescales O/H/L/C by `AdjClose/Close` so intraday features are consistent with `AdjClose`.
- QQQ's 2:1 split (March 2000) is before our window.

Reproduce with `python -m qqq.data`. Yahoo occasionally revises historical dividends, so `AdjClose` can differ slightly. **The committed file is the source of truth for every number in the report.**

## Sample periods (`src/qqq/config.py`)
| Period | Dates | Trading days | Mean daily ret | Std daily ret | Cumulative |
|---|---|---|---|---|---|
| Train | 2005-01-03 → 2018-12-31 | 3,522 | 0.050% | 1.27% | +338% |
| Validation | 2019-01-02 → 2021-12-31 | 757 | 0.140% | 1.57% | +163% |
| Test | 2022-01-03 → 2026-10-30 | 1,212 | *sealed until final run* | | |

The train period includes the 2008 crash (buy-and-hold max drawdown about −53%). Validation includes the 2020 COVID crash and rebound.
