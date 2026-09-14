"""
Sweep-and-reclaim liquidity-trap reversal strategy.

Edge premise
------------
Before a real move up, price often spikes below an obvious recent low to
trigger clustered stop-losses sitting just under it. Once that liquidity is
absorbed, waiting buyers step in and price reverses up. Entering AFTER the
sweep-and-reclaim puts you on the side of the absorbing buyers instead of
getting stopped out with the crowd that placed stops under the obvious low.
Mirror logic for shorts: a spike above an obvious recent high that gets
rejected back below it.

Intended regime
----------------
Thrives in ranging / accumulating markets and at turning points, where
liquidity pools sit under obvious lows (or over obvious highs). Dies in
strong sustained downtrends (the sweep just keeps going, there's no
absorption) and in dead low-volatility chop (the reclaim happens but there's
no follow-through to make it worth the stop distance).

Rules
-----
LONG  : current bar's low < prior bar's low (the sweep) AND current bar's
        close > prior bar's low (the reclaim). Enter next bar's open.
SHORT : current bar's high > prior bar's high (the sweep) AND current bar's
        close < prior bar's high (the reclaim). Enter next bar's open.

Stop is premise-tied: it sits just beyond the sweep candle's own extreme
(low for longs, high for shorts) plus a small ATR buffer for noise. If price
later trades back through that extreme, the "liquidity got absorbed" thesis
is falsified.

Exit/target: this strategy relies on the engine's shared chandelier ATR
trailing stop (config.ATR_TRAIL_MULT), same as every other strategy in this
system, rather than a fixed swing-high target -- it lets winners run with
the trend that follows the reclaim instead of capping them at the first
liquidity pool.

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


class SweepReclaim(Strategy):
    def __init__(
        self,
        atr_period: int = config.SWEEP_ATR_PERIOD,
        atr_stop_buffer_mult: float = config.SWEEP_ATR_STOP_BUFFER_MULT,
    ):
        self.atr_period = atr_period
        self.atr_stop_buffer_mult = atr_stop_buffer_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        high, low, close = df["high"], df["low"], df["close"]
        prev_high = high.shift(1)
        prev_low  = low.shift(1)

        atr = _atr(df, self.atr_period)

        # Sweep below the prior low, then reclaim back above it -> long
        sweep_low  = low  < prev_low
        reclaim_up = close > prev_low
        long_setup = sweep_low & reclaim_up

        # Sweep above the prior high, then reclaim back below it -> short
        sweep_high   = high > prev_high
        reclaim_down = close < prev_high
        short_setup  = sweep_high & reclaim_down

        # An inside/outside bar could technically satisfy both sides at once;
        # that's not a clean trap in either direction, so skip it.
        ambiguous = long_setup & short_setup

        raw_signal = pd.Series(0, index=df.index)
        raw_signal[long_setup & ~ambiguous]  =  1
        raw_signal[short_setup & ~ambiguous] = -1

        # Stop just beyond the sweep candle's own extreme (the level that
        # falsifies the "liquidity got absorbed" thesis if it trades back
        # through), plus a small ATR buffer for noise.
        long_stop  = low  - self.atr_stop_buffer_mult * atr
        short_stop = high + self.atr_stop_buffer_mult * atr

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
