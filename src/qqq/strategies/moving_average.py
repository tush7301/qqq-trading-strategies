"""Strategy slot 1 - Moving-average crossover. Owner: tush7301

Idea: QQQ trends. When the short-term average of price is above the long-term
average, the trend is up -> hold QQQ; otherwise step aside (or go short). The
aim is to sit out prolonged bear markets (2008, 2022) at the cost of lagging
at turning points and getting whipsawed in sideways markets.

Signal at the close of day t (AdjClose only, trailing windows only):

    ratio_t = MA_short_t / MA_long_t - 1

    ratio_t >  +band  ->  long  (1)
    ratio_t <  -band  ->  flat  (0)   or short (-1) if allow_short
    otherwise         ->  keep yesterday's position   (hysteresis)

    Before MA_long has a full window -> 0.

    Optional fast re-entry (reentry_window = R):
        also long whenever price_t > MA_R_t   (overrides flat/short)

`band` is a neutral zone around the crossover that cuts whipsaw trades;
band = 0 is the textbook crossover. short_window = 1 gives the classic
"price vs. its N-day average" rule (MA_1 = price).

`reentry_window` fixes the crossover's worst failure mode: after a fast
V-shaped crash (e.g. COVID 2020) the slow averages take months to cross back
up, so the strategy misses the rebound. With re-entry the strategy is long
if the slow trend is up OR price has recovered above a faster average. It
uses the same `ma_type`; it only switches on after the long-window warm-up.

Optional volatility cap (vol_target = T, Moreira & Muir 2017):
        position_t = signal_t * min(1, T / realized_vol_t)
    realized_vol = annualized std of the last `vol_window` daily returns
    (known at close t). Exposure is cut in turbulent markets, where crashes
    cluster; it is never levered above the signal. To avoid trading every day,
    the held size only changes when the signal itself changes, the target is
    0 or full size, or it drifts more than `vol_band` from what is held.
    vol_target="train" sets T to QQQ's realized volatility over the training
    period, computed in fit() - a target chosen without looking at any results.

Defaults = tuned configuration (notebooks/01_moving_average/tuning.ipynb):
    SMA 50/250, no band, long/flat, fast re-entry above SMA 50,
    volatility cap at QQQ's train-period volatility (~20.2%) with a 10% band.
The default needs fit(train_prices) before generate_positions (run_all, evaluate
and grid_search do this). For the plain crossover pass reentry_window=None,
vol_target=None.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qqq import features as F
from qqq.strategies.base import Strategy

MA_TYPES = ("sma", "ema")


class MovingAverageCrossover(Strategy):
    name = "ma_crossover"

    def __init__(
        self,
        short_window: int = 50,
        long_window: int = 250,
        ma_type: str = "sma",
        band: float = 0.0,
        allow_short: bool = False,
        reentry_window: int | None = 50,
        vol_target: float | str | None = "train",
        vol_window: int = 20,
        vol_band: float = 0.1,
    ):
        if not (isinstance(short_window, (int, np.integer)) and short_window >= 1):
            raise ValueError("short_window must be an integer >= 1")
        if not (isinstance(long_window, (int, np.integer)) and long_window > short_window):
            raise ValueError("long_window must be an integer > short_window")
        if ma_type not in MA_TYPES:
            raise ValueError(f"ma_type must be one of {MA_TYPES}")
        if not (0.0 <= band < 1.0):
            raise ValueError("band must be in [0, 1)")
        if reentry_window is not None and not (
            isinstance(reentry_window, (int, np.integer)) and 2 <= reentry_window < long_window
        ):
            raise ValueError("reentry_window must be None or an integer in [2, long_window)")
        self.reentry_window = None if reentry_window is None else int(reentry_window)
        if not (vol_target is None or vol_target == "train"
                or (isinstance(vol_target, (int, float)) and vol_target > 0)):
            raise ValueError("vol_target must be None, 'train' or a positive number")
        if not (isinstance(vol_window, (int, np.integer)) and vol_window >= 2):
            raise ValueError("vol_window must be an integer >= 2")
        if not (0.0 <= vol_band < 1.0):
            raise ValueError("vol_band must be in [0, 1)")
        self.vol_target = vol_target
        self.vol_window = int(vol_window)
        self.vol_band = float(vol_band)
        self.fitted_vol_target_: float | None = None
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.ma_type = ma_type
        self.band = float(band)
        self.allow_short = bool(allow_short)

    # ------------------------------------------------------------------ fit ---
    def fit(self, train_prices: pd.DataFrame) -> "MovingAverageCrossover":
        """Only needed for vol_target="train": QQQ's annualized volatility on train."""
        if self.vol_target == "train":
            r = F.daily_returns(train_prices).dropna()
            self.fitted_vol_target_ = float(r.std(ddof=1) * np.sqrt(252))
        return self

    def _resolved_vol_target(self) -> float | None:
        if self.vol_target is None:
            return None
        if self.vol_target == "train":
            if self.fitted_vol_target_ is None:
                raise ValueError('vol_target="train" requires fit(train_prices) first')
            return self.fitted_vol_target_
        return float(self.vol_target)

    # ------------------------------------------------------------------ core ---
    def _ma(self, s: pd.Series, window: int) -> pd.Series:
        if window == 1:
            return s
        return F.sma(s, window) if self.ma_type == "sma" else F.ema(s, window)

    def moving_averages(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Both averages and their ratio - handy for plots in the notebook."""
        p = prices["AdjClose"]
        short, long_ = self._ma(p, self.short_window), self._ma(p, self.long_window)
        out = pd.DataFrame({"ma_short": short, "ma_long": long_, "ratio": short / long_ - 1})
        if self.reentry_window is not None:
            out["ma_reentry"] = self._ma(p, self.reentry_window)
        return out

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        ma = self.moving_averages(prices)
        ratio = ma["ratio"]
        low = -1.0 if self.allow_short else 0.0

        signal = pd.Series(np.nan, index=prices.index)
        signal[ratio > self.band] = 1.0
        signal[ratio < -self.band] = low
        # Inside the band (and exactly at the crossover) carry the last decision
        # forward. ffill only looks backwards, so this cannot peek ahead.
        pos = signal.ffill().fillna(0.0)
        if self.reentry_window is not None:
            pos[prices["AdjClose"] > ma["ma_reentry"]] = 1.0  # NaN comparison -> False
        pos[ma["ma_long"].isna() | ma["ma_short"].isna()] = 0.0  # warm-up
        target = self._resolved_vol_target()
        if target is not None:
            pos = self._vol_cap(pos, prices, target)
        return pos.rename("position")

    def _vol_cap(self, signal: pd.Series, prices: pd.DataFrame, target: float) -> pd.Series:
        vol = F.rolling_volatility(prices, self.vol_window)  # annualized, trailing, known at t
        scale = (target / vol).clip(upper=1.0).fillna(1.0)
        desired = (signal * scale).to_numpy()
        sig = signal.to_numpy()
        out, cur, prev_sig = np.zeros(len(desired)), 0.0, 0.0
        for t in range(len(desired)):  # causal: only uses desired[t], sig[t] and past state
            x = desired[t]
            if (sig[t] != prev_sig or x == 0.0 or abs(x) == 1.0
                    or abs(x - cur) > self.vol_band):
                cur = x
            out[t], prev_sig = cur, sig[t]
        return pd.Series(out, index=signal.index)

    def params(self) -> dict:
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "ma_type": self.ma_type,
            "band": self.band,
            "allow_short": self.allow_short,
            "reentry_window": self.reentry_window,
            "vol_target": self.vol_target,
            "vol_window": self.vol_window,
            "vol_band": self.vol_band,
        }
