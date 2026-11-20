"""Future-leakage audit for every strategy (registered slots + research screen).

    python scripts/leakage_audit.py          # exits 1 if any strategy fails

Four independent checks. Positions are compared EXACTLY (no tolerance).

1. Truncation: fit on train, then positions up to date T must be identical whether
   or not the rows after T exist. Cuts inside train, at the train/validation and
   validation/test boundaries, in the COVID crash, and inside test.
2. Alternate future: replace every price after T with a different (random) future.
   Positions up to T must not change. This also catches leaks that truncation can't,
   e.g. a feature normalised with statistics of the whole series. Cuts are on or
   after the end of train, so fit() sees identical data.
3. Hidden data access: fit()/generate_positions() must not read data on their own
   (qqq.data.load_prices, pandas.read_csv). Everything must come from the frame passed in.
4. Placebo: on 20 synthetic random-walk price paths (no predictable structure), a
   leak-free strategy cannot systematically beat buy-and-hold. A strategy that peeks
   at tomorrow's return beats it massively; we flag mean Sharpe edge > 0.3 with t > 3.

Only positions are compared; no test-period performance is computed or printed.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import qqq.data  # noqa: E402
from qqq.backtest import run_backtest  # noqa: E402
from qqq.data import load_prices, split, synthetic_prices  # noqa: E402
from qqq.evaluate import PERIODS  # noqa: E402
from qqq.strategies import REGISTRY  # noqa: E402
from qqq.strategies.base import BuyAndHold, Strategy  # noqa: E402

TRUNC_CUTS = ["2010-06-30", "2018-12-31", "2020-03-16", "2021-12-31", "2024-06-28"]
FUTURE_CUTS = ["2018-12-31", "2020-03-16", "2021-12-31"]
N_PLACEBO = 20


def _candidates() -> dict[str, Strategy]:
    out = {}
    for name, cls in REGISTRY.items():
        try:
            s = cls()
            s.fit(split(load_prices(source="csv"))[0])
            s.generate_positions(load_prices(source="csv").iloc[:300])
            out[f"slot: {name}"] = cls()
        except NotImplementedError:
            continue
    from screen_strategies import screen_strategies  # noqa: E402

    for label, s in screen_strategies().items():
        if label != "Buy & hold" and not label.startswith("Slot 1"):
            out[f"screen: {label}"] = s
    return out


def _fit(strategy: Strategy, prices: pd.DataFrame) -> Strategy:
    strategy.fit(split(prices)[0])
    return strategy


def check_truncation(strategy, prices) -> str:
    _fit(strategy, prices)
    full = strategy.generate_positions(prices)
    for cut in TRUNC_CUTS:
        part = strategy.generate_positions(prices.loc[:cut])
        if not full.loc[:cut].equals(part):
            bad = (full.loc[:cut] != part).idxmax()
            return f"FAIL at cut {cut} (first diff {bad.date()})"
    return "pass"


def _alternate_future(prices: pd.DataFrame, cut: str, seed: int | None) -> pd.DataFrame:
    """Same history up to `cut`; a different future after it.
    seed=None -> MIRRORED future (actual daily returns with the sign flipped, so every
    future day moves the other way); otherwise a random walk with that seed."""
    hist = prices.loc[:cut]
    fut_idx = prices.index[prices.index > pd.Timestamp(cut)]
    if seed is None:
        actual = prices["AdjClose"].pct_change().loc[fut_idx].to_numpy()
        r = np.log1p(-np.clip(actual, -0.5, 0.5))
    else:
        r = np.random.default_rng(seed).normal(0.0, 0.02, len(fut_idx))
    rng = np.random.default_rng(seed or 0)
    last = hist.iloc[-1]
    growth = np.exp(np.cumsum(r))
    fut = pd.DataFrame(index=fut_idx)
    for col in ("Open", "High", "Low", "Close", "AdjClose"):
        fut[col] = last[col] * growth
    fut["High"] = fut[["Open", "High", "Close"]].max(axis=1) * 1.01
    fut["Low"] = fut[["Open", "Low", "Close"]].min(axis=1) * 0.99
    fut["Volume"] = last["Volume"] * rng.lognormal(0, 0.5, len(fut_idx))
    return pd.concat([hist, fut[hist.columns]])


def check_alternate_future(strategy, prices) -> str:
    _fit(strategy, prices)  # cuts are >= end of train, so the fit is identical
    base = strategy.generate_positions(prices)
    for cut in FUTURE_CUTS:
        for seed in (None, 1, 2):
            alt = strategy.generate_positions(_alternate_future(prices, cut, seed))
            if not base.loc[:cut].equals(alt.loc[:cut]):
                bad = (base.loc[:cut] != alt.loc[:cut]).idxmax()
                return f"FAIL at cut {cut} (first diff {bad.date()})"
    return "pass"


def check_hidden_data_access(strategy, prices) -> str:
    def boom(*a, **k):
        raise RuntimeError("strategy read data outside the frame it was given")

    patched = [(qqq.data, "load_prices"), (pd, "read_csv")]
    for mod in list(sys.modules.values()):
        is_strategy = getattr(mod, "__name__", "").startswith("qqq.strategies")
        if is_strategy and hasattr(mod, "load_prices"):
            patched.append((mod, "load_prices"))
    saved = [(m, a, getattr(m, a)) for m, a in patched]
    try:
        for m, a in patched:
            setattr(m, a, boom)
        strategy.fit(split(prices)[0])
        strategy.generate_positions(prices)
        return "pass"
    except RuntimeError as e:
        return f"FAIL ({e})"
    finally:
        for m, a, f in saved:
            setattr(m, a, f)


def check_placebo(strategy, prices) -> str:
    """Sharpe edge over buy-and-hold on validation dates of random-walk paths."""
    edges = []
    for seed in range(N_PLACEBO):
        fake = synthetic_prices(str(prices.index[0].date()), str(prices.index[-1].date()),
                                seed=1000 + seed)
        sharpe = {}
        for key, s in (("strat", strategy), ("bh", BuyAndHold())):
            s.fit(split(fake)[0])
            res = run_backtest(fake, s.generate_positions(fake))
            sharpe[key] = res.slice(*PERIODS["validation"]).metrics()["sharpe_ratio"]
        edges.append(sharpe["strat"] - sharpe["bh"])
    e = np.array(edges)
    t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if e.std() > 0 else 0.0
    verdict = "FAIL" if (e.mean() > 0.3 and t > 3) else "pass"
    return f"{verdict} (edge {e.mean():+.2f}, t={t:+.1f})"


def main() -> int:
    warnings.filterwarnings("ignore")
    prices = load_prices(source="csv")
    rows = {}
    for label, strat in _candidates().items():
        rows[label] = {
            "1 truncation": check_truncation(strat, prices),
            "2 alternate future": check_alternate_future(strat, prices),
            "3 hidden data access": check_hidden_data_access(strat, prices),
            "4 placebo (random walks)": check_placebo(strat, prices),
        }
        print(f"  audited {label}")

    # Positive control: a strategy that DOES peek must be caught by every check.
    class Peeker(Strategy):
        name = "peeker"

        def generate_positions(self, p):
            fwd = p["AdjClose"].shift(-1) / p["AdjClose"] - 1  # tomorrow's return
            return (fwd > 0).astype(float).rename("position")

    # Positive control 2: a z-score normalised with the FULL file's statistics.
    class FullSampleScaler(Strategy):
        name = "full_sample_scaler"

        def fit(self, train_prices):
            full = qqq.data.load_prices(source="csv")["AdjClose"].pct_change()
            self.mu, self.sd = full.mean(), full.std()
            return self

        def generate_positions(self, p):
            z = (p["AdjClose"].pct_change() - self.mu) / self.sd
            return (z.rolling(5).mean() > 0).astype(float).rename("position")

    controls = {"CONTROL: peeks at tomorrow": Peeker(),
                "CONTROL: full-sample normalisation": FullSampleScaler()}
    for label, strat in controls.items():
        rows[label] = {
            "1 truncation": check_truncation(strat, prices),
            "2 alternate future": check_alternate_future(strat, prices),
            "3 hidden data access": check_hidden_data_access(strat, prices),
            "4 placebo (random walks)": check_placebo(strat, prices),
        }
    table = pd.DataFrame(rows).T
    print("\n" + table.to_markdown())

    real = table.drop(index=list(controls))
    failed = real.apply(lambda c: c.str.startswith("FAIL")).any(axis=1)
    for label in controls:
        caught = [k for k, v in rows[label].items() if v.startswith("FAIL")]
        print(f"\n{label} -> caught by: {', '.join(caught) or 'NOTHING (audit is broken)'}")
    if failed.any():
        print(f"LEAKAGE in: {', '.join(real.index[failed])}")
        return 1
    print(f"All {len(real)} strategies pass every check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
