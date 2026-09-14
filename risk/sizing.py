"""
Strategy-agnostic position sizing and kill-switch logic.

Fixed-fractional: risk_fraction % of CURRENT equity per trade.
position_size = (equity * risk_fraction) / stop_distance
The max loss on any single trade is bounded to risk_fraction * equity by construction.
"""
import math
from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


@dataclass
class KillSwitchState:
    consecutive_losses: int = 0
    trades_today:       int = 0
    last_trade_date:    Optional[object] = None   # datetime.date
    triggered:          bool = False


def compute_position_size(
    equity:          float,
    entry_price:     float,
    stop_price:      float,
    direction:       int,           # +1 long, -1 short
    risk_fraction:   float = config.RISK_FRACTION,
    max_position_pct: float = config.MAX_POSITION_PCT,
) -> tuple[float, float]:
    """
    Returns (size_in_units, risk_amount_in_currency).

    stop_distance is always positive regardless of direction.
    If stop is on the wrong side of entry (data error), returns (0, 0).
    """
    stop_distance = direction * (entry_price - stop_price)  # must be > 0
    if stop_distance <= 0 or entry_price <= 0:
        return 0.0, 0.0

    risk_amount = equity * risk_fraction
    size = risk_amount / stop_distance

    # Hard cap: position value ≤ max_position_pct * equity
    max_size = (equity * max_position_pct) / entry_price
    size = min(size, max_size)

    size = math.floor(size * 1e8) / 1e8   # avoid floating-point overshoot
    return size, risk_amount


def check_kill_switch(
    state:       KillSwitchState,
    trade_date,                         # datetime.date of proposed trade
    last_pnl:    Optional[float] = None,
    max_consec:  Optional[int]   = config.MAX_CONSECUTIVE_LOSSES,
    max_per_day: Optional[int]   = config.MAX_TRADES_PER_DAY,
) -> bool:
    """
    Returns True if trading should be HALTED.
    Call BEFORE entering a trade; call update_kill_switch() AFTER a trade closes.
    """
    if state.triggered:
        return True

    if max_consec is not None and state.consecutive_losses >= max_consec:
        state.triggered = True
        return True

    if max_per_day is not None:
        if state.last_trade_date == trade_date:
            if state.trades_today >= max_per_day:
                return True
        # date rolled — reset counter (update happens in update_kill_switch)

    return False


def update_kill_switch(
    state:      KillSwitchState,
    trade_date,
    pnl:        float,
) -> None:
    """Update state after a trade closes."""
    if state.last_trade_date != trade_date:
        state.trades_today    = 0
        state.last_trade_date = trade_date

    state.trades_today += 1

    if pnl < 0:
        state.consecutive_losses += 1
    else:
        state.consecutive_losses = 0
