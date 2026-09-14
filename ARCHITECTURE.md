# Architecture

This document describes the module structure, the contracts between
modules, and how to extend the platform with a new strategy or instrument.

## Design principle

Every module talks to its neighbors through a narrow, typed contract
(a `DataFrame` with specific columns, a small dataclass, a tuple of two
things) rather than sharing internal state. This means:

- Strategies never see the engine, the risk module, or reporting code.
- The engine never knows which strategy produced its signals.
- The risk module never knows what timeframe or instrument it's sizing for.
- Reporting never knows how trades were generated.

The practical benefit: a new strategy is a single new file that touches
nothing else, and a new instrument is a config entry, not a code change.

## Module map

```
config.py                  Single source of truth: instruments, strategy
                            defaults, risk parameters, split ratios, paths.
                            No logic lives here, only values.

data/
  loader.py                 load(symbol, timeframe) -> OHLCV DataFrame.
                            Dispatches on instrument type (crypto -> ccxt/
                            OKX; forex -> yfinance), with ordered fallbacks
                            and CSV caching. Everything downstream only
                            ever sees the same clean, UTC-indexed frame.

strategies/
  base.py                   Strategy ABC. One abstract method:
                            generate_signals(df) -> DataFrame[signal, stop, atr].
  trend_following.py         Donchian breakout + SMA direction filter.
  mean_reversion.py           Bollinger/RSI fade.
  breakout.py                 Bollinger/Keltner volatility-squeeze breakout.
  sweep_reclaim.py             Liquidity-sweep-and-reclaim reversal (long+short).
  sweep_reclaim_long.py         Same edge, long-only variant.

engine/
  backtest.py                run(df, signals) -> (trades, equity_curve).
                            The only module that knows about bar-by-bar
                            execution order, fills, and no-lookahead
                            enforcement. Strategy-agnostic and
                            instrument-agnostic.

risk/
  sizing.py                  compute_position_size(...) — fixed-fractional
                            sizing from equity, entry, and stop distance.
                            check_kill_switch / update_kill_switch — plumbing
                            for max-consecutive-losses / max-trades-per-day
                            circuit breakers. Called by the engine; strategies
                            never size their own positions.

metrics/
  report.py                  compute_metrics / print_metrics — the full
                            performance suite from a trade list + equity
                            curve. print_regime_breakdown — two-lens
                            regime attribution. save_trade_log /
                            plot_equity_curves — artifacts. append_results_log —
                            persistent run log.

main.py                    CLI orchestration: loads data, resolves the
                            selected strategy, splits IS/OOS/holdout,
                            calls the engine, calls the reporting layer.
                            Contains no strategy logic, no sizing logic,
                            and no execution logic itself.
```

## The contracts

**`Strategy.generate_signals(df) -> signals`**
Input: an OHLCV `DataFrame` (columns `open high low close volume`, UTC
`DatetimeIndex`). Output: a `DataFrame` with the same index and exactly
three columns:

| Column | Type | Meaning |
|---|---|---|
| `signal` | `int` | `+1` long, `-1` short, `0` flat, for entry on the **next** bar |
| `stop` | `float` | Initial stop price for that signal |
| `atr` | `float` | ATR at signal time, used by the sizing module |

The shift is the strategy's responsibility, enforced by convention across
every implementation: compute indicators on bar *i*, then
`.shift(1)` the whole signals frame before returning it, so
`signals.iloc[i]` describes what to do entering bar `i+1`'s open. This
keeps the no-lookahead guarantee local to each strategy file instead of
relying on the engine to guess intent.

**`engine.backtest.run(df, signals) -> (trades, equity_curve)`**
Consumes exactly that contract. It doesn't know or care whether the
strategy was trend-following or mean-reverting — it manages one open
position at a time, evaluates stops/trailing exits at close, applies
commission and slippage on both sides of every fill, and calls into
`risk.sizing` for position size at entry. Returns a list of `Trade` records
and a per-bar equity series.

**`risk.sizing.compute_position_size(equity, entry_price, stop_price, direction) -> (size, risk_amount)`**
Pure function: no state beyond what's passed in (the kill-switch state is
threaded explicitly through `KillSwitchState`, not held globally). Fixed-
fractional: risk a constant `config.RISK_FRACTION` of current equity per
trade, capped by `config.MAX_POSITION_PCT` of equity in position value.

**`data.loader.load(symbol, timeframe) -> df`**
The only function that knows about network sources. Everything after this
call — strategies, the engine, reporting — operates on the same OHLCV
frame regardless of whether it came from OKX, a Yahoo fallback, or
yfinance.

## How a new strategy plugs in

1. Create `strategies/my_strategy.py`:

   ```python
   from strategies.base import Strategy

   class MyStrategy(Strategy):
       def __init__(self, some_param: float = config.MY_STRATEGY_PARAM):
           self.some_param = some_param

       def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
           # compute indicators on closed bars only
           # build raw_signal, stop, atr Series
           # shift all three by 1 bar before returning
           ...
   ```

2. Add its hard-coded defaults to `config.py` (grouped with the other
   strategy default blocks).

3. Register it in `main.py`:

   ```python
   from strategies.my_strategy import MyStrategy
   STRATEGIES = {..., "mystrategy": MyStrategy}
   ```

4. Run it: `python main.py --strategy mystrategy --instrument BTC/USDT`

No changes to `engine/`, `risk/`, or `metrics/` are needed — they only ever
see the `[signal, stop, atr]` contract, never the strategy's internals.

## How a new instrument plugs in

Add an entry to `config.INSTRUMENTS`:

```python
INSTRUMENTS = {
    ...
    "EUR/USDT": {"type": "crypto", "exchange": "okx", "yf_ticker": "EURUSDT"},
    "USOIL":    {"type": "forex",  "yf_ticker": "CL=F"},
}
```

`data/loader.py` dispatches on `type` — no new fetch code is required
unless the instrument needs a source neither `ccxt` nor `yfinance` covers,
in which case the change is isolated to `data/loader.py` and nothing else
in the pipeline is affected.

## CLI reference

```bash
python main.py --strategy trend --instrument XAUUSD --timeframe 1h
python main.py --strategy meanrev --instrument BTC/USDT --timeframe 4h
python main.py --strategy breakout --instrument GBPUSD --timeframe 1d
python main.py --refresh                              # force re-download instead of using the CSV cache
python main.py --strategy trend --instrument BTC/USDT --final   # also evaluates the locked holdout (one-shot)
```

| Flag | Values | Default | Meaning |
|---|---|---|---|
| `--strategy` | `trend` \| `meanrev` \| `breakout` \| `sweep` \| `sweeplong` | `trend` | Which pluggable strategy to run |
| `--instrument` | any key in `config.INSTRUMENTS` | all configured | Instrument to backtest |
| `--timeframe` | e.g. `1h`, `4h`, `1d` | `4h` | Candle interval |
| `--refresh` | flag | off | Force re-download instead of using the CSV cache |
| `--final` | flag | off | Evaluate the locked final-holdout split (one-shot) |

Outputs are written to `output/` (per-instrument trade log CSV + equity
curve PNG) and every run appends one row to `results_log.csv` in the
project root.

## Validation pipeline (`main.py` orchestration)

```
load data -> generate_signals -> split (IS 60% / OOS 20% / holdout 20%)
    -> run(IS)  -> compute_metrics -> print
    -> run(OOS) -> compute_metrics -> print          (equity carried over from IS)
    -> [only with --final] run(holdout) -> compute_metrics -> print
    -> print_regime_breakdown(dev trades, dev period)
    -> append_results_log(...)
    -> save_trade_log(...) / plot_equity_curves(...)
```

The holdout slice is computed by the same `split_df()` call every time, but
`engine.backtest.run()` is only ever invoked on it when the CLI caller
passes `--final` — the separation between "data that exists" and "data
that has been evaluated" is enforced by control flow in `main.py`, not by
withholding the data from the process at all.
