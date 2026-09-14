"""
Volatility-squeeze breakout strategy.

Detects a Bollinger-Band-inside-Keltner-Channel "squeeze" (a period of
compressed volatility) and enters in the direction of the breakout once
price clears the recent range shortly after a squeeze. Initial stop is
ATR-based; the engine's shared chandelier trailing stop (config.ATR_TRAIL_MULT)
manages the exit, same as every other strategy in this system.

All indicators use only CLOSED bars; the signal column is shift()-ed by 1
bar so entry always occurs on bar N+1's open (same convention as
strategies/trend_following.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from strategies.base import Strategy


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


class VolatilitySqueezeBreakout(Strategy):
    def __init__(
        self,
        bb_period: int = config.BREAKOUT_BB_PERIOD,
        bb_mult: float = config.BREAKOUT_BB_MULT,
        kc_period: int = config.BREAKOUT_KC_PERIOD,
        kc_mult: float = config.BREAKOUT_KC_MULT,
        range_period: int = config.BREAKOUT_RANGE_PERIOD,
        atr_period: int = config.BREAKOUT_ATR_PERIOD,
        atr_stop_mult: float = config.BREAKOUT_ATR_STOP_MULT,
        squeeze_lookback: int = 5,
    ):
        self.bb_period = bb_period
        self.bb_mult = bb_mult
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        self.range_period = range_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.squeeze_lookback = squeeze_lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close, high, low = df["close"], df["high"], df["low"]

        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        bb_upper = sma + self.bb_mult * std
        bb_lower = sma - self.bb_mult * std

        kc_atr   = _atr(df, self.kc_period)
        kc_mid   = close.ewm(span=self.kc_period, adjust=False).mean()
        kc_upper = kc_mid + self.kc_mult * kc_atr
        kc_lower = kc_mid - self.kc_mult * kc_atr

        # Squeeze: Bollinger Bands compressed INSIDE the Keltner Channel
        squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        # "recently squeezed" = squeeze was active at any point in the last
        # squeeze_lookback CLOSED bars (.shift(1) excludes the current bar)
        was_squeezed = (
            squeeze_on.shift(1).rolling(self.squeeze_lookback).max().fillna(0).astype(bool)
        )

        range_high = high.shift(1).rolling(self.range_period).max()
        range_low  = low.shift(1).rolling(self.range_period).min()

        breakout_long  = was_squeezed & (close > range_high)
        breakout_short = was_squeezed & (close < range_low)

        atr = _atr(df, self.atr_period)

        raw_signal = pd.Series(0, index=df.index)
        raw_signal[breakout_long]  =  1
        raw_signal[breakout_short] = -1

        long_stop  = close - self.atr_stop_mult * atr
        short_stop = close + self.atr_stop_mult * atr

        stop = pd.Series(np.nan, index=df.index)
        stop[raw_signal ==  1] = long_stop[raw_signal ==  1]
        stop[raw_signal == -1] = short_stop[raw_signal == -1]

        # ── Critical: shift by 1 so entry occurs on the NEXT bar ──────────────
        signals = pd.DataFrame({
            "signal": raw_signal.shift(1).fillna(0).astype(int),
            "stop":   stop.shift(1),
            "atr":    atr.shift(1),
        }, index=df.index)

        return signals
