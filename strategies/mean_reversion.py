"""
Bollinger/RSI mean-reversion (fade) strategy.

Fades price back toward the mean when it closes outside a Bollinger Band
AND RSI confirms an extreme reading. Initial stop is ATR-based; the engine's
shared chandelier trailing stop (config.ATR_TRAIL_MULT) manages the exit,
same as every other strategy in this system.

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


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


class MeanReversion(Strategy):
    def __init__(
        self,
        bb_period: int = config.MEANREV_BB_PERIOD,
        bb_std: float = config.MEANREV_BB_STD,
        rsi_period: int = config.MEANREV_RSI_PERIOD,
        rsi_oversold: float = config.MEANREV_RSI_OVERSOLD,
        rsi_overbought: float = config.MEANREV_RSI_OVERBOUGHT,
        atr_period: int = config.MEANREV_ATR_PERIOD,
        atr_stop_mult: float = config.MEANREV_ATR_STOP_MULT,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        upper_band = sma + self.bb_std * std
        lower_band = sma - self.bb_std * std

        rsi = _rsi(close, self.rsi_period)
        atr = _atr(df, self.atr_period)

        # Fade: buy when price closes below the lower band AND RSI is oversold;
        # sell when price closes above the upper band AND RSI is overbought.
        long_fade  = (close < lower_band) & (rsi < self.rsi_oversold)
        short_fade = (close > upper_band) & (rsi > self.rsi_overbought)

        raw_signal = pd.Series(0, index=df.index)
        raw_signal[long_fade]  =  1
        raw_signal[short_fade] = -1

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
