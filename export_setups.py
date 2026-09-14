"""
Exports "video-worthy" trade setups from an existing backtest run.

Read-only: this module never touches the backtest engine or strategy logic.
It reads the trade log CSVs in output/ and the cached OHLCV CSVs in
data/cache/ (both produced by main.py) and derives everything else itself.

Sweep / range model
--------------------
TrendFollowing is a Donchian-breakout strategy, not an explicit
liquidity-sweep strategy, so "sweep" here is mapped onto the breakout
event that actually triggers each trade:

  entry bar   = index[i]        (trade log's entry_time; fill at bar i open)
  sweep bar   = index[i - 1]    (the closed bar whose close broke the range)
  range window (as used by the strategy) = the DONCHIAN_PERIOD bars
                                            immediately before the sweep bar

The reported range_start/range_end are extended backward from that window
for as long as price kept respecting the same range_high/range_low levels,
so "how long has this exact level been building" becomes a real,
variable signal instead of a constant.

Note on "tp"
------------
This strategy exits via an ATR chandelier trailing stop, not a fixed
take-profit target, so there is no true TP price to report. The "tp"
field is always null; use r_multiple / outcome / exit_time instead.

CLI
---
    python export_setups.py --pair ETH --top 10
    python export_setups.py --pair BTC --top 5 --include-losses
    python export_setups.py --top 10                    # all configured symbols
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config

# ── Scoring constants ────────────────────────────────────────────────────────
DEFAULT_TOP_N            = 10
R_CAP                    = 5.0                       # r_multiple >= this saturates the R sub-score at 100
SWEEP_ATR_CAP             = 2.0                       # wick beyond the level >= this many ATRs saturates sweep score
DURATION_BONUS_BARS       = config.DONCHIAN_PERIOD * 2  # extra contained bars (beyond the min window) for full bonus
CONTAINMENT_SENSITIVITY   = 3.0                       # how harshly choppy in-range candles are penalized

W_RANGE       = 0.20   # (a) how clean the pre-sweep range was
W_SWEEP       = 0.20   # (b) how decisive the sweep was
W_CLEAN_RUN   = 0.30   # (c) how clean the post-entry run was
W_R_MULTIPLE  = 0.30   # (d) final R multiple


# ── Indicator helper (standalone; not imported from strategies/) ─────────────

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ── Path / symbol helpers ─────────────────────────────────────────────────────

def normalize_symbol(pair: str) -> str:
    """'ETH' / 'eth' / 'ETH/USDT' -> 'ETH/USDT'."""
    pair = pair.strip().upper().replace(" ", "")
    if "/" in pair:
        return pair
    base = pair[:-4] if pair.endswith("USDT") else pair
    return f"{base}/USDT"


def _trade_log_path(symbol: str) -> Path:
    return config.OUTPUT_DIR / f"trade_log_{symbol.replace('/', '_')}.csv"


def _cache_path(symbol: str, timeframe: str) -> Path:
    return config.CACHE_DIR / f"{symbol.replace('/', '_')}_{timeframe}.csv"


def load_trade_log(symbol: str) -> pd.DataFrame:
    path = _trade_log_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"no trade log for {symbol} at {path} (run main.py first)")
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"], utc=True)
    return df


def load_ohlcv_cache(symbol: str, timeframe: str) -> pd.DataFrame:
    path = _cache_path(symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"no cached OHLCV for {symbol} {timeframe} at {path} (run main.py first)")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _locate_bar(ohlcv_index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    pos = ohlcv_index.get_indexer([ts])[0]
    if pos == -1:
        pos = ohlcv_index.get_indexer([ts], method="nearest")[0]
    return int(pos)


# ── Per-trade scoring ──────────────────────────────────────────────────────

def score_trade(row: pd.Series, ohlcv: pd.DataFrame, timeframe: str, symbol: str) -> dict | None:
    """Returns a setup dict, or None if the trade can't be fairly scored
    (not enough history before it, or degenerate range/ATR)."""
    high, low, idx = ohlcv["high"].values, ohlcv["low"].values, ohlcv.index

    direction = 1 if row["direction"] == "LONG" else -1
    entry_idx = _locate_bar(idx, row["entry_time"])
    exit_idx  = _locate_bar(idx, row["exit_time"])

    sweep_idx       = entry_idx - 1
    range_start_idx = sweep_idx - config.DONCHIAN_PERIOD
    range_end_idx   = sweep_idx - 1

    if range_start_idx < 0 or sweep_idx < 1 or exit_idx <= entry_idx:
        return None

    range_high = float(high[range_start_idx:range_end_idx + 1].max())
    range_low  = float(low[range_start_idx:range_end_idx + 1].min())
    if not np.isfinite(range_high) or not np.isfinite(range_low) or range_high <= range_low:
        return None

    # (a) how clean the pre-sweep range was ------------------------------------
    # Extend backward from the trigger window for as long as price kept
    # respecting these exact levels, so duration becomes a real variable
    # instead of the fixed DONCHIAN_PERIOD.
    contained_start_idx = range_end_idx
    j = range_end_idx
    while j >= 1 and high[j] <= range_high and low[j] >= range_low:
        contained_start_idx = j
        j -= 1
    duration_bars = range_end_idx - contained_start_idx + 1

    window_hi = high[contained_start_idx:range_end_idx + 1]
    window_lo = low[contained_start_idx:range_end_idx + 1]
    avg_bar_range_pct = float(np.mean(window_hi - window_lo)) / (range_high - range_low)

    duration_score    = 100 * min(1.0, max(0.0, (duration_bars - config.DONCHIAN_PERIOD) / DURATION_BONUS_BARS))
    containment_score = 100 * min(1.0, max(0.0, 1 - avg_bar_range_pct * CONTAINMENT_SENSITIVITY))
    range_score        = 0.5 * duration_score + 0.5 * containment_score

    # (b) how decisive the sweep was ---------------------------------------
    atr_at_sweep = _atr(ohlcv, config.ATR_PERIOD).values[sweep_idx]
    if not np.isfinite(atr_at_sweep) or atr_at_sweep <= 0:
        return None

    if direction == 1:
        sweep_price    = float(high[sweep_idx])
        sweep_dist_atr = max(0.0, (sweep_price - range_high) / atr_at_sweep)
    else:
        sweep_price    = float(low[sweep_idx])
        sweep_dist_atr = max(0.0, (range_low - sweep_price) / atr_at_sweep)

    sweep_score = 100 * min(1.0, sweep_dist_atr / SWEEP_ATR_CAP)

    # (c) how clean the post-entry run was ----------------------------------
    entry_price     = float(row["entry_price"])
    sl              = float(row["initial_stop"])
    risk_per_unit   = abs(entry_price - sl)
    if risk_per_unit <= 0:
        return None

    path_hi = high[entry_idx:exit_idx + 1]
    path_lo = low[entry_idx:exit_idx + 1]
    if direction == 1:
        mae = max(0.0, entry_price - float(path_lo.min()))
    else:
        mae = max(0.0, float(path_hi.max()) - entry_price)
    mae_r = mae / risk_per_unit

    r_multiple = float(row["r_multiple"])
    run_efficiency  = r_multiple / (r_multiple + mae_r) if r_multiple > 0 else 0.0
    clean_run_score = 100 * max(0.0, min(1.0, run_efficiency))

    # (d) final R multiple ---------------------------------------------------
    r_score = 100 * min(1.0, max(0.0, r_multiple / R_CAP))

    total_score = (
        W_RANGE * range_score
        + W_SWEEP * sweep_score
        + W_CLEAN_RUN * clean_run_score
        + W_R_MULTIPLE * r_score
    )

    outcome = "win" if r_multiple > 0 else ("loss" if r_multiple < 0 else "breakeven")
    setup_id = (
        f"{symbol.replace('/', '')}_{timeframe}_"
        f"{row['entry_time'].strftime('%Y%m%dT%H%M')}_{row['direction']}"
    )

    return {
        "pair":        symbol,
        "timeframe":   timeframe,
        "setup_id":    setup_id,
        "score":       round(total_score, 1),
        "range_start": idx[contained_start_idx].isoformat(),
        "range_end":   idx[range_end_idx].isoformat(),
        "range_high":  round(range_high, 6),
        "range_low":   round(range_low, 6),
        "sweep_time":  idx[sweep_idx].isoformat(),
        "sweep_price": round(sweep_price, 6),
        "entry_time":  row["entry_time"].isoformat(),
        "entry_price": round(entry_price, 6),
        "sl":          round(sl, 6),
        "tp":          None,   # strategy trails ATR-stop; no fixed TP exists
        "exit_time":   row["exit_time"].isoformat(),
        "outcome":     outcome,
        "r_multiple":  round(r_multiple, 3),
        "score_breakdown": {
            "range_cleanliness":      round(range_score, 1),
            "sweep_decisiveness":     round(sweep_score, 1),
            "post_entry_cleanliness": round(clean_run_score, 1),
            "r_multiple_score":       round(r_score, 1),
        },
    }


# ── Per-symbol pipeline ────────────────────────────────────────────────────

def build_setups(symbol: str, timeframe: str, top_n: int, include_losses: bool) -> tuple[list[dict], int]:
    trades = load_trade_log(symbol)
    ohlcv  = load_ohlcv_cache(symbol, timeframe)

    if not include_losses:
        trades = trades[trades["r_multiple"] > 0]

    setups, skipped = [], 0
    for _, row in trades.iterrows():
        s = score_trade(row, ohlcv, timeframe, symbol)
        if s is None:
            skipped += 1
        else:
            setups.append(s)

    setups.sort(key=lambda s: s["score"], reverse=True)
    return setups[:top_n], skipped


def export(symbol: str, timeframe: str, top_n: int, include_losses: bool, output_dir: Path) -> list[dict]:
    setups, skipped = build_setups(symbol, timeframe, top_n, include_losses)

    output_dir.mkdir(parents=True, exist_ok=True)
    for s in setups:
        (output_dir / f"{s['setup_id']}.json").write_text(json.dumps(s, indent=2))

    print(f"  {symbol} {timeframe}: exported {len(setups)} setup(s), skipped {skipped} unscorable -> {output_dir}")
    for s in setups:
        print(f"    {s['score']:5.1f}  {s['setup_id']:<40} {s['outcome']:<5} {s['r_multiple']:+.2f}R")
    return setups


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export top video-worthy setups from existing backtest outputs (read-only)."
    )
    parser.add_argument("--pair", type=str, default=None,
                        help="e.g. ETH, BTC, ETH/USDT. Defaults to all symbols in config.SYMBOLS.")
    parser.add_argument("--timeframe", type=str, default=config.TIMEFRAME)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="Top N setups per pair.")
    parser.add_argument("--include-losses", action="store_true",
                        help="Also score and export losing trades (for loss-breakdown videos).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Defaults to <project>/setups")
    args = parser.parse_args()

    symbols    = [normalize_symbol(args.pair)] if args.pair else list(config.SYMBOLS)
    output_dir = Path(args.output_dir) if args.output_dir else (config.ROOT_DIR / "setups")

    print("=" * 60)
    print("  SETUP EXPORTER  (video-worthiness scoring, read-only)")
    print(f"  Timeframe : {args.timeframe}")
    print(f"  Top N     : {args.top} per pair")
    print(f"  Losses    : {'included' if args.include_losses else 'excluded'}")
    print("=" * 60)

    for symbol in symbols:
        try:
            export(symbol, args.timeframe, args.top, args.include_losses, output_dir)
        except FileNotFoundError as e:
            print(f"  Skipping {symbol}: {e}")

    print("\n  Note: 'tp' is always null -- this strategy exits via an ATR trailing")
    print("  stop, not a fixed take-profit target.")


if __name__ == "__main__":
    main()
