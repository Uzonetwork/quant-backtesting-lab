"""
Bar-by-bar backtest engine.

Design invariants
─────────────────
• Signal on bar N  →  entry at bar N open  (signals already shifted by strategy).
• Stop and trail are evaluated at bar N close (conservative: no intrabar stops).
• Costs applied at entry AND exit (commission + slippage, both sides).
• One position per instrument at a time.
• Equity used for sizing is equity AT THE OPEN of the entry bar (no lookahead).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from risk.sizing import (
    KillSwitchState,
    check_kill_switch,
    compute_position_size,
    update_kill_switch,
)


# ── Trade record ───────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time:  object
    exit_time:   object
    direction:   int        # +1 / -1
    entry_price: float
    exit_price:  float
    size:        float
    stop_price:  float      # initial stop at entry
    pnl_gross:   float      # before costs
    pnl_net:     float      # after costs
    cost:        float
    r_multiple:  float      # pnl_net / initial_risk
    equity_after: float
    exit_reason: str        # "stop", "trail", "end_of_data"


# ── Cost helper ────────────────────────────────────────────────────────────────

def _fill_price(price: float, direction: int, side: str) -> float:
    """Apply slippage: buy fills higher, sell fills lower."""
    slip = config.SLIPPAGE_PCT
    if side == "entry":
        return price * (1 + direction * slip)
    else:   # exit
        return price * (1 - direction * slip)


def _commission(price: float, size: float) -> float:
    return price * size * config.COMMISSION_PCT


# ── Main engine ────────────────────────────────────────────────────────────────

def run(
    df:       pd.DataFrame,           # OHLCV, clean, UTC index
    signals:  pd.DataFrame,           # [signal, stop, atr] — same index, already shifted
    label:    str = "backtest",
) -> tuple[list[Trade], pd.Series]:
    """
    Returns
    -------
    trades      : list[Trade]
    equity_curve: pd.Series indexed same as df (equity at close of each bar)
    """
    equity    = config.INITIAL_EQUITY
    ks_state  = KillSwitchState()
    trades: list[Trade] = []

    # Per-bar equity curve (value at close of bar)
    eq_index  = df.index
    eq_values = np.full(len(df), np.nan)
    eq_values[0] = equity

    # Open position state
    in_position     = False
    direction       = 0
    entry_price     = 0.0
    entry_fill      = 0.0
    size            = 0.0
    initial_stop    = 0.0
    initial_risk    = 0.0
    trail_stop      = 0.0
    highest_since   = -np.inf    # for long trail
    lowest_since    =  np.inf    # for short trail
    entry_bar_idx   = 0
    entry_time      = None

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    opens  = df["open"].values
    index  = df.index

    sig    = signals["signal"].values
    stops  = signals["stop"].values
    atrs   = signals["atr"].values

    for i in range(1, len(df)):
        bar_date = index[i].date()

        # ── Manage open position ───────────────────────────────────────────────
        if in_position:
            # Update trail (ratchet only in profitable direction)
            atr_trail = atrs[i] * config.ATR_TRAIL_MULT if not np.isnan(atrs[i]) else atrs[entry_bar_idx] * config.ATR_TRAIL_MULT

            if direction == 1:
                highest_since = max(highest_since, high[i])
                new_trail     = highest_since - atr_trail
                trail_stop    = max(trail_stop, new_trail)
                hit_stop      = close[i] < trail_stop or close[i] < initial_stop
            else:
                lowest_since  = min(lowest_since, low[i])
                new_trail     = lowest_since + atr_trail
                trail_stop    = min(trail_stop, new_trail)
                hit_stop      = close[i] > trail_stop or close[i] > initial_stop

            exit_reason = None
            if hit_stop:
                exit_reason = "trail" if (
                    (direction == 1 and trail_stop > initial_stop) or
                    (direction == -1 and trail_stop < initial_stop)
                ) else "stop"

            if exit_reason or i == len(df) - 1:
                exit_reason = exit_reason or "end_of_data"
                exit_fill   = _fill_price(close[i], direction, "exit")
                cost_exit   = _commission(exit_fill, size)
                cost_total  = cost_exit  # entry cost already deducted

                pnl_gross = direction * (exit_fill - entry_fill) * size
                pnl_net   = pnl_gross - cost_total

                equity   += pnl_net
                r_mult    = pnl_net / initial_risk if initial_risk > 0 else 0.0

                trades.append(Trade(
                    entry_time   = entry_time,
                    exit_time    = index[i],
                    direction    = direction,
                    entry_price  = entry_fill,
                    exit_price   = exit_fill,
                    size         = size,
                    stop_price   = initial_stop,
                    pnl_gross    = pnl_gross,
                    pnl_net      = pnl_net,
                    cost         = cost_exit,
                    r_multiple   = r_mult,
                    equity_after = equity,
                    exit_reason  = exit_reason,
                ))

                update_kill_switch(ks_state, bar_date, pnl_net)
                in_position = False

        # ── Check for new entry ────────────────────────────────────────────────
        if not in_position and sig[i] != 0:
            if check_kill_switch(ks_state, bar_date):
                eq_values[i] = equity
                continue

            signal_dir  = int(sig[i])
            stop_price  = stops[i]
            atr_val     = atrs[i]

            if np.isnan(stop_price) or np.isnan(atr_val) or atr_val <= 0:
                eq_values[i] = equity
                continue

            # Entry fills at this bar's open (signal was on prior bar's close)
            raw_entry   = opens[i]
            entry_fill  = _fill_price(raw_entry, signal_dir, "entry")
            cost_entry  = _commission(entry_fill, 1.0)   # placeholder, updated below

            sz, risk_amt = compute_position_size(
                equity      = equity,
                entry_price = entry_fill,
                stop_price  = stop_price,
                direction   = signal_dir,
            )

            if sz <= 0:
                eq_values[i] = equity
                continue

            cost_entry  = _commission(entry_fill, sz)
            equity     -= cost_entry            # deduct entry cost immediately

            in_position   = True
            direction     = signal_dir
            entry_price   = raw_entry
            size          = sz
            initial_stop  = stop_price
            initial_risk  = risk_amt
            trail_stop    = stop_price          # trail starts at initial stop
            entry_time    = index[i]
            entry_bar_idx = i

            if direction == 1:
                highest_since = high[i]
            else:
                lowest_since  = low[i]

        eq_values[i] = equity

    # Forward-fill equity for bars with open position (value doesn't change mid-trade)
    eq_series = pd.Series(eq_values, index=eq_index)
    eq_series = eq_series.ffill()
    return trades, eq_series
