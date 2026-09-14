"""
Entry point. Runs in-sample / out-of-sample (and optionally the locked final
holdout) backtest for one strategy on one or more instruments.

Usage:
    python main.py                                          # trend, all configured instruments
    python main.py --strategy trend --instrument XAUUSD --timeframe 1h
    python main.py --strategy meanrev --instrument BTC/USDT --refresh
    python main.py --strategy breakout --instrument GBPUSD --timeframe 1d
    python main.py --final                                  # ALSO evaluates the locked holdout split (ONE-SHOT)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config
from data.loader import load
from strategies.trend_following import TrendFollowing
from strategies.mean_reversion import MeanReversion
from strategies.breakout import VolatilitySqueezeBreakout
from strategies.sweep_reclaim import SweepReclaim
from strategies.sweep_reclaim_long import SweepReclaimLong
from engine.backtest import run
from metrics.report import (
    compute_metrics,
    print_metrics,
    print_regime_breakdown,
    save_trade_log,
    plot_equity_curves,
    append_results_log,
)

STRATEGIES = {
    "trend":     TrendFollowing,
    "meanrev":   MeanReversion,
    "breakout":  VolatilitySqueezeBreakout,
    "sweep":     SweepReclaim,
    "sweeplong": SweepReclaimLong,
}

BARS_PER_YEAR = {"1h": 8766, "4h": 2190, "1d": 365, "1w": 52}


def split_df(df: pd.DataFrame):
    """
    Chronological IS / OOS / FINAL-HOLDOUT split (config.IS_RATIO / OOS_RATIO /
    HOLDOUT_RATIO). The holdout slice is only ever fed into the backtest engine
    when the caller passes --final -- otherwise it is sliced out here but never
    touched again.
    """
    n       = len(df)
    is_end  = int(n * config.IS_RATIO)
    oos_end = int(n * (config.IS_RATIO + config.OOS_RATIO))
    return df.iloc[:is_end], df.iloc[is_end:oos_end], df.iloc[oos_end:]


def backtest_symbol(
    symbol:        str,
    strategy_name: str,
    timeframe:     str,
    force_refresh: bool = False,
    final:         bool = False,
) -> None:
    print(f"\n{'-'*60}")
    print(f"  Symbol    : {symbol}")
    print(f"  Strategy  : {strategy_name}")
    print(f"  Timeframe : {timeframe}")
    print(f"{'-'*60}")

    df = load(symbol, timeframe=timeframe, force_refresh=force_refresh)

    strategy_cls = STRATEGIES[strategy_name]
    strategy     = strategy_cls()
    signals      = strategy.generate_signals(df)

    is_df,  oos_df,  holdout_df  = split_df(df)
    is_sig, oos_sig, holdout_sig = split_df(signals)

    print(f"  In-sample      : {is_df.index[0].date()} to {is_df.index[-1].date()}  ({len(is_df):,} bars)")
    print(f"  Out-of-sample  : {oos_df.index[0].date()} to {oos_df.index[-1].date()}  ({len(oos_df):,} bars)")
    if final:
        print(f"  FINAL HOLDOUT  : {holdout_df.index[0].date()} to {holdout_df.index[-1].date()}"
              f"  ({len(holdout_df):,} bars)  -- EVALUATING NOW (ONE-SHOT) --")
    else:
        print(f"  Final holdout  : {holdout_df.index[0].date()} to {holdout_df.index[-1].date()}"
              f"  ({len(holdout_df):,} bars)  [LOCKED -- not evaluated; re-run with --final for the one-shot test]")

    # ── In-sample run ─────────────────────────────────────────────────────────
    is_trades, is_equity = run(is_df, is_sig, label="in-sample")

    # ── Out-of-sample run ─────────────────────────────────────────────────────
    # Start OOS from IS ending equity so the curve is continuous
    oos_start_eq   = is_equity.iloc[-1]
    original_eq    = config.INITIAL_EQUITY
    config.INITIAL_EQUITY = oos_start_eq
    oos_trades, oos_equity = run(oos_df, oos_sig, label="out-of-sample")
    config.INITIAL_EQUITY = original_eq

    bars_per_year = BARS_PER_YEAR.get(timeframe, 2190)

    # ── Standard IS / OOS metrics ─────────────────────────────────────────────
    is_metrics  = compute_metrics(is_trades,  is_equity,  f"IN-SAMPLE  [{symbol}/{strategy_name}]", bars_per_year)
    oos_metrics = compute_metrics(oos_trades, oos_equity, f"OUT-OF-SAMPLE [{symbol}/{strategy_name}]", bars_per_year)
    print_metrics(is_metrics)
    print_metrics(oos_metrics)

    dev_trades     = is_trades + oos_trades
    dev_df         = pd.concat([is_df, oos_df])
    periods_label  = "IS + OOS combined"
    holdout_trades: list = []
    holdout_equity_out = None
    final_metrics       = None

    # ── Final holdout (ONE-SHOT, --final only) ────────────────────────────────
    if final:
        print("\n" + "!" * 60)
        print("  FINAL HOLDOUT EVALUATION -- THIS IS A ONE-SHOT TEST.")
        print("  If these numbers disappoint you, DO NOT tweak parameters and")
        print("  re-run --final. That defeats the entire purpose of a holdout.")
        print("!" * 60)

        holdout_start_eq = oos_equity.iloc[-1]
        original_eq = config.INITIAL_EQUITY
        config.INITIAL_EQUITY = holdout_start_eq
        holdout_trades, holdout_equity = run(holdout_df, holdout_sig, label="final-holdout")
        config.INITIAL_EQUITY = original_eq

        final_metrics = compute_metrics(
            holdout_trades, holdout_equity,
            f"FINAL HOLDOUT (ONE-SHOT) [{symbol}/{strategy_name}]", bars_per_year,
        )
        print_metrics(final_metrics)

        dev_trades    = dev_trades + holdout_trades
        dev_df        = df
        periods_label = "IS + OOS + FINAL HOLDOUT"
        holdout_equity_out = holdout_equity

        print("!" * 60)
        print("  END OF ONE-SHOT FINAL HOLDOUT EVALUATION.")
        print("!" * 60)

    # ── Regime breakdown ──────────────────────────────────────────────────────
    print_regime_breakdown(dev_trades, dev_df, symbol, periods_label=periods_label, timeframe=timeframe)

    # ── Results log (anti-data-mining: count every configuration you test) ────
    n_logged = append_results_log(
        strategy_name = strategy_name,
        symbol         = symbol,
        timeframe      = timeframe,
        params         = dict(vars(strategy)),
        oos_metrics    = oos_metrics,
        final_metrics  = final_metrics,
    )
    print(f"\n  Strategies tested so far: {n_logged}. "
          f"Remember that ~1 in 20 will look good by chance.")
    print(f"  Full log -> {config.RESULTS_LOG_PATH}")

    # ── Outputs ───────────────────────────────────────────────────────────────
    save_trade_log(is_trades, oos_trades, symbol, holdout_trades=holdout_trades or None)
    plot_equity_curves(
        is_equity, oos_equity, symbol,
        df=dev_df, holdout_equity=holdout_equity_out, strategy_name=strategy_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-instrument, multi-strategy backtesting lab")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default=config.DEFAULT_STRATEGY,
                        help="Strategy to run (default: %(default)s)")
    parser.add_argument("--instrument", default=None,
                        help=f"One of {list(config.INSTRUMENTS)}. Default: run all configured instruments.")
    parser.add_argument("--timeframe", default=config.TIMEFRAME,
                        help="Candle interval, e.g. 1h, 4h, 1d (default: %(default)s)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download data (required after switching exchange/source)")
    parser.add_argument("--final", action="store_true",
                        help="Evaluate the locked final-holdout split. ONE-SHOT -- see README.")
    args = parser.parse_args()

    if args.instrument and args.instrument not in config.INSTRUMENTS:
        print(f"Unknown instrument '{args.instrument}'. Configured: {list(config.INSTRUMENTS)}")
        sys.exit(1)

    instruments = [args.instrument] if args.instrument else list(config.INSTRUMENTS.keys())

    print("\n" + "=" * 60)
    print("  MULTI-INSTRUMENT / MULTI-STRATEGY BACKTEST LAB")
    print(f"  Strategy   : {args.strategy}")
    print(f"  Instruments: {', '.join(instruments)}")
    print(f"  Timeframe  : {args.timeframe}")
    print(f"  Risk/trade : {config.RISK_FRACTION*100:.2f}%  |  "
          f"Costs: {config.COMMISSION_PCT*100:.2f}%+{config.SLIPPAGE_PCT*100:.2f}% per side")
    if args.final:
        print("  !! --final flag set: locked holdout WILL be evaluated (one-shot) !!")
    print("=" * 60)

    for instrument in instruments:
        backtest_symbol(instrument, args.strategy, args.timeframe,
                         force_refresh=args.refresh, final=args.final)

    print(f"\n  Output files in: {config.OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
