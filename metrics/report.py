"""
Computes and prints all performance metrics.
Generates equity curve PNG (with regime shading) and trade-log CSV.
Strategy-agnostic: operates only on trades list + equity curve.
All console strings are ASCII-only (Windows cp1252 safe).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.backtest import Trade


# ── Core metric computations ──────────────────────────────────────────────────

def _cagr(start_equity: float, end_equity: float, n_bars: int, bars_per_year: float) -> float:
    if n_bars <= 0 or start_equity <= 0:
        return 0.0
    years = n_bars / bars_per_year
    return (end_equity / start_equity) ** (1 / years) - 1 if years > 0 else 0.0


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Returns (max_drawdown_pct, max_drawdown_duration_bars)."""
    roll_max  = equity.cummax()
    drawdown  = (equity - roll_max) / roll_max
    max_dd    = drawdown.min()
    underwater = (drawdown < 0).astype(int)
    streaks, dur = [], 0
    for v in underwater:
        if v:
            dur += 1
        else:
            if dur:
                streaks.append(dur)
            dur = 0
    if dur:
        streaks.append(dur)
    max_dur = max(streaks) if streaks else 0
    return max_dd, max_dur


def _sharpe(returns: pd.Series, bars_per_year: float) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(bars_per_year)


def _sortino(returns: pd.Series, bars_per_year: float) -> float:
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return (returns.mean() / downside.std()) * np.sqrt(bars_per_year)


def _longest_losing_streak(trades: list[Trade]) -> int:
    streak = best = 0
    for t in trades:
        if t.pnl_net < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


# ── Full metrics block ────────────────────────────────────────────────────────

def compute_metrics(
    trades:        list[Trade],
    equity:        pd.Series,
    label:         str,
    bars_per_year: float = 2190.0,
) -> dict:
    if not trades:
        return {"label": label, "n_trades": 0}

    pnls    = np.array([t.pnl_net for t in trades])
    r_mults = np.array([t.r_multiple for t in trades])
    wins    = pnls[pnls > 0]
    losses  = pnls[pnls < 0]

    win_rate     = len(wins) / len(pnls)
    avg_win      = wins.mean()   if len(wins)   else 0.0
    avg_loss     = losses.mean() if len(losses) else 0.0
    win_loss_r   = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf

    pos_r = r_mults[r_mults > 0]
    neg_r = r_mults[r_mults < 0]
    expectancy_r = (
        win_rate * pos_r.mean() - (1 - win_rate) * abs(neg_r.mean())
        if len(pos_r) and len(neg_r) else r_mults.mean()
    )

    start_eq  = equity.iloc[0]
    end_eq    = equity.iloc[-1]
    total_ret = (end_eq - start_eq) / start_eq
    n_bars    = len(equity)
    cagr      = _cagr(start_eq, end_eq, n_bars, bars_per_year)

    bar_returns = equity.pct_change().dropna()
    sharpe      = _sharpe(bar_returns, bars_per_year)
    sortino     = _sortino(bar_returns, bars_per_year)
    max_dd, max_dd_dur = _max_drawdown(equity)
    ll_streak   = _longest_losing_streak(trades)

    return {
        "label":              label,
        "n_trades":           len(trades),
        "total_return_pct":   total_ret * 100,
        "cagr_pct":           cagr * 100,
        "win_rate_pct":       win_rate * 100,
        "avg_win":            avg_win,
        "avg_loss":           avg_loss,
        "win_loss_ratio":     win_loss_r,
        "avg_r":              r_mults.mean(),
        "expectancy_r":       expectancy_r,
        "profit_factor":      profit_factor,
        "max_dd_pct":         max_dd * 100,
        "max_dd_dur_bars":    max_dd_dur,
        "sharpe":             sharpe,
        "sortino":            sortino,
        "longest_loss_streak": ll_streak,
        "start_equity":       start_eq,
        "end_equity":         end_eq,
    }


def print_metrics(m: dict) -> None:
    sep = "=" * 60
    if m.get("n_trades", 0) == 0:
        print(f"\n{sep}\n{m['label'].upper()}: No trades.\n{sep}")
        return
    print(f"\n{sep}")
    print(f"  {m['label'].upper()}")
    print(sep)
    print(f"  Trades          : {m['n_trades']}")
    print(f"  Total Return    : {m['total_return_pct']:+.2f}%")
    print(f"  CAGR            : {m['cagr_pct']:+.2f}%")
    print(f"  Win Rate        : {m['win_rate_pct']:.1f}%")
    print(f"  Avg Win         : ${m['avg_win']:,.2f}")
    print(f"  Avg Loss        : ${m['avg_loss']:,.2f}")
    print(f"  Win/Loss Ratio  : {m['win_loss_ratio']:.2f}x")
    print(f"  Avg R / trade   : {m['avg_r']:.3f}R")
    print(f"  Expectancy      : {m['expectancy_r']:.3f}R")
    print(f"  Profit Factor   : {m['profit_factor']:.2f}")
    print(f"  Max Drawdown    : {m['max_dd_pct']:.2f}%  ({m['max_dd_dur_bars']} bars)")
    print(f"  Sharpe          : {m['sharpe']:.2f}")
    print(f"  Sortino         : {m['sortino']:.2f}")
    print(f"  Longest L-streak: {m['longest_loss_streak']}")
    print(f"  Start Equity    : ${m['start_equity']:,.2f}")
    print(f"  End Equity      : ${m['end_equity']:,.2f}")
    print(sep)


# ── Regime analysis ───────────────────────────────────────────────────────────

def _trade_mini_metrics(trades: list[Trade]) -> dict:
    """Condensed stats for a trade sub-group (no equity series needed)."""
    if not trades:
        return {"n": 0}
    pnls   = np.array([t.pnl_net for t in trades])
    r_mult = np.array([t.r_multiple for t in trades])
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf     = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    pos_r  = r_mult[r_mult > 0]
    neg_r  = r_mult[r_mult < 0]
    wr     = len(wins) / len(pnls)
    exp    = (
        wr * pos_r.mean() - (1 - wr) * abs(neg_r.mean())
        if len(pos_r) and len(neg_r) else r_mult.mean()
    )
    return {
        "n":             len(trades),
        "win_rate":      wr * 100,
        "expectancy":    exp,
        "profit_factor": pf,
        "total_pnl":     pnls.sum(),
        "avg_r":         r_mult.mean(),
    }


def _lookup_regime(trade: Trade, regime_series: pd.Series) -> str:
    """Find the regime value at (or just before) the trade entry time."""
    idx = regime_series.index.searchsorted(trade.entry_time, side="right") - 1
    if idx < 0:
        idx = 0
    return regime_series.iloc[idx]


def print_regime_breakdown(
    all_trades:    list[Trade],
    df:            pd.DataFrame,   # full OHLCV covering the reported periods
    symbol:        str,
    periods_label: str = "IS + OOS combined",
    timeframe:     str = config.TIMEFRAME,
) -> None:
    """
    Two-lens regime breakdown:

    Lens 1 -- SMA filter (same definition the strategy uses):
      BULL bars = close > SMA(SMA_PERIOD)  -> strategy only enters LONGS
      BEAR bars = close < SMA(SMA_PERIOD)  -> strategy only enters SHORTS
      So LONG trade performance = bull-regime performance, and vice versa.

    Lens 2 -- Price drawdown from rolling peak:
      Normal     : drawdown from 200-bar high < 15%
      Correction : drawdown 15-30%
      Crash      : drawdown > 30%
    This cuts ACROSS direction and reveals how the strategy handles large price dislocations.
    """
    if not all_trades:
        print("\n  No trades to compute regime breakdown.")
        return

    close     = df["close"]
    sma       = close.rolling(config.SMA_PERIOD).mean()
    peak      = close.rolling(config.SMA_PERIOD, min_periods=1).max()
    dd_pct    = (close - peak) / peak * 100

    # Per-bar SMA regime (used for background shading reference)
    sma_bull = (close >= sma).fillna(True)   # pre-SMA-warmup bars default to neutral/bull
    bull_bar_pct = sma_bull.mean() * 100
    bear_bar_pct = 100 - bull_bar_pct

    # Drawdown regime series
    dd_regime = pd.Series("Normal", index=df.index, dtype=object)
    dd_regime[dd_pct < -15] = "Correction"
    dd_regime[dd_pct < -30] = "Crash"

    # Split trades by direction (= SMA regime at entry, by construction of the strategy)
    long_trades  = [t for t in all_trades if t.direction ==  1]
    short_trades = [t for t in all_trades if t.direction == -1]

    # Split trades by price-drawdown regime at entry
    dd_groups: dict[str, list[Trade]] = {"Normal": [], "Correction": [], "Crash": []}
    for t in all_trades:
        dd_groups[_lookup_regime(t, dd_regime)].append(t)

    sep  = "=" * 60
    sep2 = "-" * 60

    print(f"\n{sep}")
    print(f"  REGIME BREAKDOWN  [{symbol}]  ({periods_label})")
    print(sep)

    # ── Market time ──────────────────────────────────────────────────────────
    total_bars = len(df)
    print(f"\n  Market time in each regime (SMA-{config.SMA_PERIOD} on {timeframe} bars):")
    print(f"    BULL (close >= SMA) : {bull_bar_pct:.1f}% of bars")
    print(f"    BEAR (close <  SMA) : {bear_bar_pct:.1f}% of bars")
    print(f"    Total bars          : {total_bars:,}")

    # ── Lens 1: Direction / SMA regime ───────────────────────────────────────
    print(f"\n{sep2}")
    print("  LENS 1: Direction = SMA-regime  (LONG trades = bull, SHORT = bear)")
    print(sep2)

    for label, trades, note in [
        ("BULL regime -- LONG trades   (close > SMA at entry)", long_trades,  "strategy buys breakouts above SMA"),
        ("BEAR regime -- SHORT trades  (close < SMA at entry)", short_trades, "strategy shorts breakdowns below SMA"),
    ]:
        m = _trade_mini_metrics(trades)
        print(f"\n  {label}:")
        if m["n"] == 0:
            print("    No trades.")
            continue
        pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "inf"
        print(f"    Trades       : {m['n']}")
        print(f"    Win Rate     : {m['win_rate']:.1f}%")
        print(f"    Expectancy   : {m['expectancy']:+.3f}R")
        print(f"    Profit Factor: {pf_str}")
        print(f"    Total P&L    : ${m['total_pnl']:,.0f}")

    # ── Lens 2: Price drawdown at entry ──────────────────────────────────────
    print(f"\n{sep2}")
    print("  LENS 2: Price drawdown from rolling peak at trade entry")
    print(f"  (rolling {config.SMA_PERIOD}-bar high on {timeframe} close)")
    print(sep2)

    max_dd_seen = dd_pct.min()
    print(f"\n  Maximum price drawdown in full dataset : {max_dd_seen:.1f}%")
    print(f"  {'Regime':<28}  {'Trades':>6}  {'WR%':>5}  {'Exp':>7}  {'PF':>5}  {'P&L':>10}")
    print(f"  {'-'*28}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*10}")

    for regime_name, threshold_note in [
        ("Normal   (< 15% from peak)", ""),
        ("Correction (-15% to -30%)", ""),
        ("Crash      (>  30% from peak)", ""),
    ]:
        key = regime_name.split("(")[0].strip().split()[0]   # "Normal", "Correction", "Crash"
        trades_in = dd_groups.get(key, [])
        m = _trade_mini_metrics(trades_in)
        if m["n"] == 0:
            print(f"  {regime_name:<28}  {'0':>6}  {'--':>5}  {'--':>7}  {'--':>5}  {'--':>10}")
            continue
        pf_str  = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "  inf"
        pnl_str = f"${m['total_pnl']:,.0f}"
        print(f"  {regime_name:<28}  {m['n']:>6}  {m['win_rate']:>4.0f}%  "
              f"{m['expectancy']:>+6.3f}R  {pf_str:>5}  {pnl_str:>10}")

    # ── Interpretation hint ──────────────────────────────────────────────────
    long_m  = _trade_mini_metrics(long_trades)
    short_m = _trade_mini_metrics(short_trades)
    print(f"\n  Quick read:")
    if short_m["n"] == 0:
        print("    No short trades -- the test window contained no bear-regime bars")
        print("    (or insufficient bars for SMA to warm up before regime ended).")
        print("    This means the test window was ENTIRELY a bull market.")
        print("    The strategy has NO demonstrated bear-market behaviour in this data.")
    elif short_m.get("profit_factor", 0) >= 1.0 and short_m.get("expectancy", -1) > 0:
        print("    Short-side (bear regime) has positive expectancy and PF >= 1.")
        print("    The strategy shows some ability to profit in downtrends.")
    else:
        print("    Short-side (bear regime) expectancy or PF is sub-1.")
        print("    The strategy struggled or had insufficient bear-regime data.")
        print("    Do NOT conclude the strategy has a short-side edge from this window.")

    print(f"\n{sep}")


# ── Trade log CSV ─────────────────────────────────────────────────────────────

def save_trade_log(
    is_trades:      list[Trade],
    oos_trades:     list[Trade],
    symbol:         str,
    holdout_trades: Optional[list[Trade]] = None,
    path:           Optional[Path] = None,
) -> Path:
    if path is None:
        safe = symbol.replace("/", "_")
        path = config.OUTPUT_DIR / f"trade_log_{safe}.csv"

    holdout_trades = holdout_trades or []
    is_set  = set(id(t) for t in is_trades)
    oos_set = set(id(t) for t in oos_trades)

    rows = []
    for t in is_trades + oos_trades + holdout_trades:
        period = "IS" if id(t) in is_set else ("OOS" if id(t) in oos_set else "HOLDOUT")
        rows.append({
            "entry_time":   t.entry_time,
            "exit_time":    t.exit_time,
            "direction":    "LONG" if t.direction == 1 else "SHORT",
            "entry_price":  round(t.entry_price, 4),
            "exit_price":   round(t.exit_price, 4),
            "size":         round(t.size, 8),
            "initial_stop": round(t.stop_price, 4),
            "r_multiple":   round(t.r_multiple, 3),
            "pnl_gross":    round(t.pnl_gross, 2),
            "pnl_net":      round(t.pnl_net, 2),
            "cost":         round(t.cost, 2),
            "equity_after": round(t.equity_after, 2),
            "exit_reason":  t.exit_reason,
            "period":       period,
        })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Trade log saved -> {path}")
    return path


# ── Results log (anti-data-mining) ────────────────────────────────────────────

def append_results_log(
    strategy_name: str,
    symbol:        str,
    timeframe:     str,
    params:        dict,
    oos_metrics:   dict,
    final_metrics: Optional[dict] = None,
    path:          Optional[Path] = None,
) -> int:
    """
    Appends one row per run to the results log CSV. This is the honest defense
    against data mining: count your attempts. Returns the total number of runs
    logged so far (including this one).
    """
    if path is None:
        path = config.RESULTS_LOG_PATH

    def _num(m: Optional[dict], key: str):
        if not m:
            return ""
        v = m.get(key)
        return round(v, 4) if isinstance(v, (int, float)) else ""

    row = {
        "run_time_utc":            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy":                strategy_name,
        "instrument":              symbol,
        "timeframe":               timeframe,
        "params":                  json.dumps(params, default=str, sort_keys=True),
        "oos_trades":              oos_metrics.get("n_trades", 0),
        "oos_total_return_pct":    _num(oos_metrics, "total_return_pct"),
        "oos_cagr_pct":            _num(oos_metrics, "cagr_pct"),
        "oos_sharpe":              _num(oos_metrics, "sharpe"),
        "oos_profit_factor":       _num(oos_metrics, "profit_factor"),
        "oos_max_dd_pct":          _num(oos_metrics, "max_dd_pct"),
        "oos_expectancy_r":        _num(oos_metrics, "expectancy_r"),
        "final_holdout_evaluated": final_metrics is not None,
        "final_total_return_pct":  _num(final_metrics, "total_return_pct"),
    }

    header_needed = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header_needed, index=False)

    with open(path, "r", encoding="utf-8") as f:
        n_rows = sum(1 for _ in f) - 1   # minus header row
    return max(n_rows, 0)


# ── Equity curve plot ─────────────────────────────────────────────────────────

def plot_equity_curves(
    is_equity:      pd.Series,
    oos_equity:     pd.Series,
    symbol:         str,
    df:             Optional[pd.DataFrame] = None,   # full OHLCV for price panel + regime shading
    holdout_equity: Optional[pd.Series] = None,       # only passed when run with --final
    strategy_name:  str = "",
    path:           Optional[Path] = None,
) -> Path:
    if path is None:
        safe = symbol.replace("/", "_")
        path = config.OUTPUT_DIR / f"equity_{safe}.png"

    has_price   = df is not None and len(df) > 0
    has_holdout = holdout_equity is not None and len(holdout_equity) > 0

    if has_price:
        fig, (ax_eq, ax_px) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        )
    else:
        fig, ax_eq = plt.subplots(figsize=(14, 6))
        ax_px = None

    # ── Equity panel ─────────────────────────────────────────────────────────
    ax_eq.plot(is_equity.index,  is_equity.values,  color="#2196F3", lw=1.5, label="In-sample",     zorder=3)
    ax_eq.plot(oos_equity.index, oos_equity.values, color="#FF5722", lw=1.5, label="Out-of-sample", zorder=3)
    if has_holdout:
        ax_eq.plot(holdout_equity.index, holdout_equity.values, color="#9C27B0", lw=1.5,
                   label="Final holdout (one-shot)", zorder=3)

    split_date = oos_equity.index[0]
    ax_eq.axvline(split_date, color="gray", linestyle="--", lw=1,
                  label=f"IS/OOS split ({split_date.date()})", zorder=4)
    if has_holdout:
        holdout_split_date = holdout_equity.index[0]
        ax_eq.axvline(holdout_split_date, color="purple", linestyle=":", lw=1,
                      label=f"OOS/Holdout split ({holdout_split_date.date()})", zorder=4)

    # Bear-regime shading on equity panel
    if has_price:
        equity_parts = [is_equity, oos_equity] + ([holdout_equity] if has_holdout else [])
        full_equity = pd.concat(equity_parts)
        full_equity = full_equity[~full_equity.index.duplicated(keep="last")]
        sma   = df["close"].rolling(config.SMA_PERIOD).mean()
        is_bear = df["close"] < sma

        # Find contiguous bear stretches and shade them
        bear_starts = []
        in_bear = False
        for ts, flag in is_bear.items():
            if flag and not in_bear:
                bear_starts.append((ts, None))
                in_bear = True
            elif not flag and in_bear:
                bear_starts[-1] = (bear_starts[-1][0], ts)
                in_bear = False
        if in_bear:
            bear_starts[-1] = (bear_starts[-1][0], df.index[-1])

        for (start, end) in bear_starts:
            ax_eq.axvspan(start, end, alpha=0.08, color="red", zorder=1)

        bear_patch = mpatches.Patch(color="red", alpha=0.2, label="Bear regime (close < SMA)")
        handles, labels = ax_eq.get_legend_handles_labels()
        ax_eq.legend(handles + [bear_patch], labels + ["Bear regime (close < SMA)"],
                     loc="upper left", fontsize=8)
    else:
        ax_eq.legend(loc="upper left", fontsize=9)

    title_suffix = f"  ({strategy_name})" if strategy_name else ""
    ax_eq.set_title(f"{symbol} -- Equity Curve{title_suffix}", fontsize=12, fontweight="bold")
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # OOS drawdown on secondary y-axis
    ax2 = ax_eq.twinx()
    oos_roll_max = oos_equity.cummax()
    dd_oos = (oos_equity - oos_roll_max) / oos_roll_max * 100
    ax2.fill_between(oos_equity.index, dd_oos, 0, alpha=0.12, color="#FF5722")
    ax2.set_ylabel("OOS DD%", color="#FF5722", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#FF5722", labelsize=7)
    ax2.set_ylim(-100, 5)

    # ── Price + SMA panel ─────────────────────────────────────────────────────
    if has_price and ax_px is not None:
        full_idx = full_equity.index
        price_slice = df["close"].reindex(full_idx, method="ffill")
        sma_slice   = df["close"].rolling(config.SMA_PERIOD).mean().reindex(full_idx, method="ffill")

        ax_px.plot(price_slice.index, price_slice.values, color="#607D8B", lw=0.8, label="Price")
        ax_px.plot(sma_slice.index,   sma_slice.values,   color="#FFC107", lw=1.0,
                   label=f"SMA({config.SMA_PERIOD})")
        ax_px.fill_between(price_slice.index,
                            price_slice.values, sma_slice.values,
                            where=(price_slice.values < sma_slice.values),
                            alpha=0.12, color="red", interpolate=True)
        ax_px.axvline(split_date, color="gray", linestyle="--", lw=1)
        ax_px.set_ylabel("Price ($)")
        ax_px.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_px.legend(loc="upper left", fontsize=7)
        ax_px.set_xlabel("Date")
    else:
        ax_eq.set_xlabel("Date")

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Equity curve saved -> {path}")
    return path
