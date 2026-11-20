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

`band` is a neutral zone around the crossover that cuts whipsaw trades;
band = 0 is the textbook crossover. short_window = 1 gives the classic
"price vs. its N-day average" rule (MA_1 = price).

Tuning workflow: notebooks/01_moving_average/ (grid on train, choose on validation).
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
        long_window: int = 200,
        ma_type: str = "sma",
        band: float = 0.0,
        allow_short: bool = False,
    ):
        if not (isinstance(short_window, (int, np.integer)) and short_window >= 1):
            raise ValueError("short_window must be an integer >= 1")
        if not (isinstance(long_window, (int, np.integer)) and long_window > short_window):
            raise ValueError("long_window must be an integer > short_window")
        if ma_type not in MA_TYPES:
            raise ValueError(f"ma_type must be one of {MA_TYPES}")
        if not (0.0 <= band < 1.0):
            raise ValueError("band must be in [0, 1)")
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.ma_type = ma_type
        self.band = float(band)
        self.allow_short = bool(allow_short)

    # ------------------------------------------------------------------ core ---
    def _ma(self, s: pd.Series, window: int) -> pd.Series:
        if window == 1:
            return s
        return F.sma(s, window) if self.ma_type == "sma" else F.ema(s, window)

    def moving_averages(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Both averages and their ratio - handy for plots in the notebook."""
        p = prices["AdjClose"]
        short, long_ = self._ma(p, self.short_window), self._ma(p, self.long_window)
        return pd.DataFrame({"ma_short": short, "ma_long": long_, "ratio": short / long_ - 1})

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
        pos[ma["ma_long"].isna() | ma["ma_short"].isna()] = 0.0  # warm-up
        return pos.rename("position")

    def params(self) -> dict:
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "ma_type": self.ma_type,
            "band": self.band,
            "allow_short": self.allow_short,
        }
