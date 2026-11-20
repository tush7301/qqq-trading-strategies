"""Shared project configuration.

FROZEN CONTRACT: change only via a PR approved by all workstream owners.
Every workstream reads dates/paths from here so results line up at collation.
"""
from pathlib import Path

TICKER = "QQQ"  # Invesco QQQ Trust (Nasdaq-100)

# ---- Sample periods (inclusive) --------------------------------------------
# Train:      fit models / pick parameters.
# Validation: compare candidate parameters / architectures (no refitting on it
#             for the final model choice beyond selection).
# Test:       touched ONCE, at collation, for the numbers in the final report.
DATA_START = "2005-01-01"
TRAIN_END = "2018-12-31"
VALIDATION_START = "2019-01-01"
VALIDATION_END = "2021-12-31"
TEST_START = "2022-01-01"
DATA_END = "2026-10-30"  # last trading day in the committed data.

# ---- Backtest conventions --------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.0  # fallback only, when no daily T-bill series is passed to metrics
# Uninvested capital earns the 13-week US T-bill rate (data/raw/IRX.csv), and
# Sharpe/Sortino are measured on returns in EXCESS of that rate (textbook definition).
CASH_EARNS_RISK_FREE = True
RISK_FREE_TICKER = "^IRX"  # CBOE 13-week T-bill yield index, in percent
TRANSACTION_COST_BPS = 5.0   # one-way cost per unit of turnover, in basis points.
RANDOM_SEED = 42

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
POSITIONS_DIR = RESULTS / "positions"
METRICS_DIR = RESULTS / "metrics"
FIGURES_DIR = RESULTS / "figures"
PRICES_CSV = DATA_RAW / f"{TICKER}.csv"
RISK_FREE_CSV = DATA_RAW / "IRX.csv"
