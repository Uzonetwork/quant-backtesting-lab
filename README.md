# Quantitative Backtesting & Strategy-Validation Platform

*Designed and built solo, end-to-end, as a demonstration of how I approach
data-heavy engineering problems: multi-source ingestion pipelines, a clean
and extensible architecture, and statistical rigor built into the pipeline
rather than bolted on afterward.*

*The same architecture — pluggable components behind narrow contracts,
automatic multi-source data ingestion, a validation pipeline designed to
resist fooling itself — applies just as well to any data-heavy application;
quantitative finance is simply where it's demonstrated here.*

A modular research platform for developing, backtesting, and rigorously
validating systematic trading strategies across crypto and forex markets.
Built to demonstrate production-grade engineering practices applied to
quantitative finance tooling: clean separation of concerns, a pluggable
strategy interface, realistic execution modeling, and built-in statistical
safeguards against overfitting.

This is not a single strategy with a hardcoded backtest — it's a small
**framework**. Any strategy that implements one interface gets, for free:
realistic fixed-fractional position sizing, transaction-cost-aware
execution, a chronological in-sample / out-of-sample / locked-holdout
validation pipeline, multi-lens regime analysis, and a persistent audit log
of every configuration ever tested.

## What it does

- **Ingests OHLCV market data** from multiple asset classes and sources
  through a single interface, with automatic fallback chains and CSV
  caching.
- **Runs any registered strategy** through a shared, bar-by-bar execution
  engine that enforces no-lookahead, applies commission/slippage, and sizes
  every position with fixed-fractional risk.
- **Validates out-of-sample** with a chronological three-way split
  (in-sample / out-of-sample / a final holdout that stays untouched until
  explicitly unlocked) instead of a single train/test split.
- **Breaks results down by market regime** (trend direction and
  drawdown-from-peak severity) so a strategy's performance can be
  attributed to the conditions it actually saw, not treated as one
  monolithic number.
- **Logs every run** — strategy, instrument, timeframe, parameters, and
  resulting metrics — to a durable CSV, so the number of configurations
  tested is always visible rather than lost to memory.

## Architecture

```
                    ┌─────────────────┐
                    │     main.py      │   CLI: --strategy --instrument
                    │  (orchestration) │        --timeframe --refresh --final
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌──────────────────┐
│  data/loader   │   │   strategies/    │   │  engine/backtest  │
│                │   │                  │   │                   │
│ multi-source   │   │ pluggable        │   │ bar-by-bar loop,  │
│ ingestion +    │──▶│ Strategy         │──▶│ no-lookahead,     │
│ CSV caching    │   │ interface        │   │ cost modeling     │
└───────────────┘   └──────────────────┘   └─────────┬─────────┘
                                                       │
                                             ┌─────────▼─────────┐
                                             │   risk/sizing      │
                                             │ fixed-fractional   │
                                             │ position sizing +  │
                                             │ kill-switch        │
                                             └─────────┬─────────┘
                                                       │
                                             ┌─────────▼─────────┐
                                             │  metrics/report     │
                                             │ IS/OOS/holdout      │
                                             │ metrics, regime      │
                                             │ breakdown, results   │
                                             │ log, equity plots    │
                                             └──────────────────┘
```

**Data layer** (`data/loader.py`) — dispatches on instrument type rather
than hardcoding one venue. Crypto pairs go through `ccxt`, using OKX's
backward-pagination candle endpoint as the primary source (with generic
forward-pagination `ccxt` and a raw Yahoo Finance v8 API call as ordered
fallbacks); forex/futures instruments are sourced through `yfinance` and
resampled to the target timeframe. Every fetch path converges on the same
clean, UTC-indexed OHLCV `DataFrame` contract and is cached to CSV so
repeated runs don't re-hit the network. Fetches print their date span and
flag anything statistically too short to validate on.

**Strategy layer** (`strategies/`) — a one-method abstract interface
(`Strategy.generate_signals(df) -> DataFrame[signal, stop, atr]`). Every
strategy is fully vectorized, computes indicators only on closed bars, and
shifts its signal forward one bar before returning it, so the interface
itself makes lookahead bias structurally difficult to introduce. Multiple
strategy archetypes are implemented against the same interface (trend
breakout, mean-reversion, volatility-squeeze breakout, liquidity-sweep
reversal), each swappable via a single CLI flag with zero changes to the
engine, risk module, or reporting layer.

**Execution engine** (`engine/backtest.py`) — a single bar-by-bar loop
shared by every strategy. Entries fill at the bar *after* the signal,
stops and trailing exits are evaluated at close (no intrabar assumptions),
and commission + slippage are applied on both sides of every trade. Equity
for position sizing is always the equity at the open of the entry bar —
no forward-looking equity curve is ever available to the sizing logic.

**Risk layer** (`risk/sizing.py`) — fixed-fractional position sizing
(risk a constant percentage of current equity per trade, sized from the
stop distance), with a hard position-value cap and optional kill-switch
plumbing (max consecutive losses, max trades/day).

**Validation & reporting** (`metrics/report.py`) — computes the full
performance-metric suite (CAGR, Sharpe, Sortino, max drawdown, profit
factor, expectancy, etc.) independently for in-sample, out-of-sample, and
(when explicitly requested) the final holdout period; runs a two-lens
regime breakdown (trend-direction and price-drawdown severity) over
realized trades; and appends a row to a persistent results log every
single run, so the platform itself tracks how many configurations have
been evaluated.

## Anti-overfitting design

Backtesting frameworks are easy to fool yourself with. This one bakes in
three structural guards rather than leaving them to discipline:

1. **Three-way chronological split** — in-sample (60%) / out-of-sample
   (20%) / final holdout (20%). The holdout is only ever fed into the
   backtest engine behind an explicit `--final` flag; ordinary development
   runs never touch it.
2. **One-shot holdout semantics** — evaluating the holdout prints an
   explicit warning framing it as a single, final check rather than
   something to iterate against.
3. **A persistent, append-only results log** — every run (strategy,
   instrument, timeframe, parameters, and resulting metrics) is written to
   `results_log.csv`, and the report surfaces how many configurations have
   been tested so far, since that count is the actual signal for how much
   to trust an apparently good result.

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data ingestion | `ccxt` (crypto exchange APIs), `yfinance` (forex/futures), `requests`/`curl_cffi` (HTTP fallbacks) |
| Data handling | `pandas`, `numpy` |
| Visualization | `matplotlib` |
| Interface | `argparse` CLI |
| Persistence | CSV (data cache, trade logs, results log) |

No external database, no notebook dependency, no proprietary data vendor —
runs anywhere Python runs.

## Getting started

```bash
git clone [REPO URL]
cd trend_system
python -m venv venv && venv\Scripts\activate   # Windows; `source venv/bin/activate` elsewhere
pip install -r requirements.txt
python main.py --strategy trend --instrument BTC/USDT --timeframe 4h
```

That one command fetches/caches data, runs the in-sample and out-of-sample
backtests, prints a regime breakdown, and writes a trade log, an equity
curve, and a results-log entry to disk. Every argument shown is swappable
— other strategies, instruments, and timeframes are registered the same
way. Full CLI reference in [ARCHITECTURE.md](ARCHITECTURE.md).

## Extending the platform

- **New strategy**: implement `strategies/base.py`'s `Strategy` interface
  in a new file, register it under a short CLI name in `main.py`. No other
  module changes required.
- **New instrument**: add an entry to `config.INSTRUMENTS` specifying its
  asset type and data source. The loader dispatches automatically.

Full details in [ARCHITECTURE.md](ARCHITECTURE.md).
