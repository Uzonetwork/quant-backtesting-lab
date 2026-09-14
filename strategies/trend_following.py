"""
Donchian-breakout trend-following strategy (Turtle lineage).

Signal generation is FULLY vectorized and shift()-ed so the engine
can never peek at the current bar's close for the current bar's entry.

All indicators are computed on CLOSED bars only.
The signal column is shifted forward by 1 bar before returning,
so signal[i] = "enter on bar i+1 open."
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from strategies.base import Strategy


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


class TrendFollowing(Strategy):
    def __init__(
        self,
        donchian_period: int = config.DONCHIAN_PERIOD,
        sma_period: int = config.SMA_PERIOD,
        atr_period: int = config.ATR_PERIOD,
        atr_stop_mult: float = config.ATR_STOP_MULT,
    ):
        self.donchian_period = donchian_period
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        sma = close.rolling(self.sma_period).mean()
        atr = _atr(df, self.atr_period)

        # Donchian channel: use .shift(1) to exclude the current bar
        # so the breakout is confirmed on a CLOSED candle
        don_high = high.shift(1).rolling(self.donchian_period).max()
        don_low  = low.shift(1).rolling(self.donchian_period).min()

        bull_filter = close > sma
        bear_filter = close < sma

        long_breakout  = close > don_high
        short_breakout = close < don_low

        raw_signal = pd.Series(0, index=df.index)
        raw_signal[bull_filter & long_breakout]   =  1
        raw_signal[bear_filter & short_breakout]  = -1

        # Initial stop price at signal bar
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
