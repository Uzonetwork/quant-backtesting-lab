#!/usr/bin/env python3
"""
fetch_history_colab.py
======================
Run on Google Colab to pull full multi-year 4H OHLCV history for
BTC/USDT and ETH/USDT.

Exchange priority : Bybit -> OKX  (whichever the machine can reach)
Output format     : Identical to trend_system/data/cache/ CSVs
                    (index="timestamp" UTC, columns=open,high,low,close,volume)
Drop the output CSVs into trend_system/data/cache/ and run main.py
with no --refresh flag.

Usage (in a Colab cell):
    !python fetch_history_colab.py
    -- or --
    %run fetch_history_colab.py

Pagination strategies
---------------------
Bybit  : forward, since-cursor from 2017-01-01
         /v5/market/kline, max 200 bars/request

OKX    : BACKWARD from now toward 2020-01-01
         OKX's 'after' param means "return bars OLDER than this timestamp"
         so forward pagination via 'since' always returns zero bars.
         We call publicGetMarketHistoryCandles directly (bypasses ccxt's
         broken parameter remapping) and page backward in 100-bar chunks.
"""

# ── 0. Auto-install ────────────────────────────────────────────────────────────
import subprocess, sys

def _pip(*pkgs):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *pkgs],
        stdout=subprocess.DEVNULL,
    )

try:
    import ccxt
    if int(ccxt.__version__.split(".")[0]) < 4:
        raise ImportError("old version")
except (ImportError, Exception):
    print("Installing ccxt >= 4 ...")
    _pip("ccxt>=4.2.0")
    import ccxt

try:
    import pandas as pd
except ImportError:
    _pip("pandas")
    import pandas as pd

# ── 1. Configuration ──────────────────────────────────────────────────────────
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

SYMBOLS   = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "4h"
MIN_SLEEP = 0.20   # seconds; exchange.rateLimit is also respected

# Bybit: go back as far as available (their data starts ~2019-2020)
BYBIT_FETCH_FROM = datetime(2017, 1, 1, tzinfo=timezone.utc)
BYBIT_LIMIT      = 200   # V5 /v5/market/kline max

# OKX: backward pagination floor — don't go older than this
OKX_FLOOR_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
OKX_LIMIT      = 100    # /api/v5/market/history-candles max

# Exchanges to try, in order. No limit needed here (each fetch fn uses its own).
EXCHANGE_CHAIN = ["bybit", "okx"]

# Minimum acceptable span for a saved CSV to be considered valid
MIN_SPAN_YEARS = 2.0

# ── 2. CSV format helpers ─────────────────────────────────────────────────────

def _cache_filename(symbol: str, timeframe: str) -> str:
    """Replicates trend_system/data/loader._cache_path naming."""
    return symbol.replace("/", "_") + f"_{timeframe}.csv"


def _bars_to_df(bars: list) -> pd.DataFrame:
    """
    Convert raw [[ts_ms, o, h, l, c, v], ...] to a DataFrame whose CSV
    format exactly matches trend_system/data/loader.py:
      index name  : "timestamp"
      index dtype : datetime64[UTC]  (written as "2021-01-01 00:00:00+00:00")
      columns     : open, high, low, close, volume  (float64)
    """
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


# ── 3a. Bybit — forward pagination ────────────────────────────────────────────

def _fetch_bybit(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Pages forward from BYBIT_FETCH_FROM using the standard since-cursor.
    Bybit V5 honours the 'since' parameter correctly.
    """
    since_ms  = int(BYBIT_FETCH_FROM.timestamp() * 1000)
    sleep_s   = max(exchange.rateLimit / 1000, MIN_SLEEP)
    all_bars: list = []

    print(f"    Bybit: forward pagination from {BYBIT_FETCH_FROM.date()} "
          f"({BYBIT_LIMIT} bars/req) ...")

    while True:
        try:
            bars = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=BYBIT_LIMIT
            )
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            print(f"    Transient error ({e}), retrying in 5s ...")
            time.sleep(5)
            bars = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=BYBIT_LIMIT
            )

        if not bars:
            break

        new_bars = [b for b in bars if b[0] >= since_ms]
        if not new_bars:
            break

        all_bars.extend(new_bars)
        last_ts  = new_bars[-1][0]
        since_ms = last_ts + 1

        if len(all_bars) % 2000 < BYBIT_LIMIT or len(bars) < BYBIT_LIMIT:
            dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"    ... {len(all_bars):,} bars  (up to {dt})")

        if len(bars) < BYBIT_LIMIT:
            break

        time.sleep(sleep_s)

    if not all_bars:
        raise RuntimeError(f"Bybit returned no bars for {symbol}")

    df = _bars_to_df(all_bars)
    print(f"    Done: {len(df):,} bars  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ── 3b. OKX — backward pagination ────────────────────────────────────────────

def _fetch_okx(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    OKX requires backward pagination.

    OKX V5 /api/v5/market/history-candles semantics:
      'after'  param = return bars with timestamp OLDER than this value
                       (confusingly named — think of it as the right-hand
                       boundary of the window, exclusive)
      'before' param = return bars with timestamp NEWER than this value

    Response is always DESC order (newest row first).
    We start from the present and walk backward until OKX_FLOOR_DATE,
    using the oldest timestamp in each batch as the next 'after' cursor.

    We call exchange.publicGetMarketHistoryCandles() directly to avoid
    ccxt's fetch_ohlcv remapping 'since' onto 'after' with inverted
    semantics (which is why the forward-pagination path returns zero bars).
    """
    floor_ms = int(OKX_FLOOR_DATE.timestamp() * 1000)
    sleep_s  = max(exchange.rateLimit / 1000, MIN_SLEEP)

    # Translate to OKX-native identifiers
    market   = exchange.market(symbol)
    inst_id  = market["id"]                           # e.g. "BTC-USDT"
    bar_code = exchange.timeframes.get(timeframe)     # e.g. "4H"
    if bar_code is None:
        raise ValueError(f"Timeframe '{timeframe}' not in OKX timeframes map")

    all_bars: list = []
    after_cursor: int | None = None   # None on first call = start from now

    print(f"    OKX: backward pagination {inst_id} {bar_code}, "
          f"floor={OKX_FLOOR_DATE.date()} ({OKX_LIMIT} bars/req) ...")

    while True:
        req: dict = {
            "instId": inst_id,
            "bar":    bar_code,
            "limit":  str(OKX_LIMIT),
        }
        if after_cursor is not None:
            req["after"] = str(after_cursor)

        try:
            response = exchange.publicGetMarketHistoryCandles(req)
        except AttributeError:
            # Fallback: ccxt renamed the implicit method in some versions
            response = exchange.public_get_market_history_candles(req)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            print(f"    Transient error ({e}), retrying in 5s ...")
            time.sleep(5)
            response = exchange.publicGetMarketHistoryCandles(req)

        rows = response.get("data", [])
        if not rows:
            print("    OKX returned empty data — no more history.")
            break

        # rows[0] = newest, rows[-1] = oldest (DESC order from OKX)
        oldest_ts_in_batch = int(rows[-1][0])

        batch: list = []
        for row in rows:
            ts = int(row[0])
            if ts < floor_ms:
                continue
            try:
                batch.append([
                    ts,
                    float(row[1]),   # open
                    float(row[2]),   # high
                    float(row[3]),   # low
                    float(row[4]),   # close
                    float(row[5]),   # volume (base currency)
                ])
            except (ValueError, IndexError):
                continue

        all_bars.extend(batch)

        dt = datetime.fromtimestamp(
            oldest_ts_in_batch / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")
        print(f"    ... {len(all_bars):,} bars  (oldest in batch: {dt})")

        # Next page: bars older than the oldest we just received
        after_cursor = oldest_ts_in_batch

        if oldest_ts_in_batch <= floor_ms:
            print(f"    Reached floor date {OKX_FLOOR_DATE.date()}, stopping.")
            break
        if len(rows) < OKX_LIMIT:
            print(f"    Got {len(rows)} < {OKX_LIMIT} rows — no more history.")
            break

        time.sleep(sleep_s)

    if not all_bars:
        raise RuntimeError(f"OKX returned no usable bars for {symbol}")

    df = _bars_to_df(all_bars)   # sorts ASC and dedupes
    print(f"    Done: {len(df):,} bars  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ── 4. Exchange dispatcher ────────────────────────────────────────────────────

_FETCH_FN = {
    "bybit": _fetch_bybit,
    "okx":   _fetch_okx,
}


def _probe_and_fetch(exchange_id: str, symbol: str, timeframe: str) -> "pd.DataFrame | None":
    """
    Instantiate exchange, verify connectivity and symbol, dispatch to the
    exchange-specific fetch function.  Returns None on any failure.
    """
    fetch_fn = _FETCH_FN.get(exchange_id)
    if fetch_fn is None:
        print(f"  No fetch strategy defined for '{exchange_id}' — skipping.")
        return None

    try:
        exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})

        print(f"  Connecting to {exchange_id} ...")
        exchange.load_markets()

        available = exchange.markets
        used_symbol = symbol
        if symbol not in available:
            # Some exchanges list USDT-margined perps as BTC/USDT:USDT
            alt = symbol + ":USDT"
            if alt in available:
                print(f"    {symbol} not in spot markets; falling back to perp {alt}")
                used_symbol = alt
            else:
                print(f"    {symbol} not available on {exchange_id} — skipping.")
                return None

        print(f"  {exchange_id}: {used_symbol} found.")
        df = fetch_fn(exchange, used_symbol, timeframe)
        return df

    except ccxt.NetworkError as e:
        print(f"  {exchange_id}: network error — {e}")
        return None
    except ccxt.ExchangeError as e:
        print(f"  {exchange_id}: exchange error — {e}")
        return None
    except RuntimeError as e:
        print(f"  {exchange_id}: {e}")
        return None
    except Exception as e:
        print(f"  {exchange_id}: unexpected error — {type(e).__name__}: {e}")
        return None


# ── 5. Span validation ────────────────────────────────────────────────────────

def _validate_span(filename: str, df: pd.DataFrame) -> bool:
    """
    Returns True if the CSV covers at least MIN_SPAN_YEARS.
    Prints a WARNING and returns False if not.
    This is NOT advisory — the caller must exit non-zero on False.
    """
    span_days  = (df.index[-1] - df.index[0]).days
    span_years = span_days / 365.25
    if span_years < MIN_SPAN_YEARS:
        print()
        print("!" * 62)
        print(f"  WARNING: {filename}")
        print(f"  Span is only {span_years:.1f} years ({span_days} days).")
        print(f"  Required minimum: {MIN_SPAN_YEARS:.0f} years.")
        print()
        print("  This file is TOO SHORT for reliable backtesting.")
        print("  The IS/OOS split and regime breakdown will both be")
        print("  statistically meaningless with this little data.")
        print()
        print("  DO NOT copy this file into trend_system/data/cache/.")
        print("  Try a different exchange or network and re-run.")
        print("!" * 62)
        return False
    return True


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 62)
    print("  Crypto 4H History Fetcher  (trend_system compatible)")
    print(f"  Symbols   : {', '.join(SYMBOLS)}")
    print(f"  Timeframe : {TIMEFRAME}")
    print(f"  Exchanges : {' -> '.join(EXCHANGE_CHAIN)}")
    print(f"  Min span  : {MIN_SPAN_YEARS:.0f} years (hard check)")
    print("=" * 62)

    saved_files:  list[str] = []
    failed_files: list[str] = []

    for symbol in SYMBOLS:
        print(f"\n{'─' * 62}")
        print(f"  {symbol}")
        print(f"{'─' * 62}")

        df           = None
        used_exchange = None

        for exchange_id in EXCHANGE_CHAIN:
            print(f"\n  [Trying {exchange_id}]")
            result = _probe_and_fetch(exchange_id, symbol, TIMEFRAME)
            if result is not None and not result.empty:
                df            = result
                used_exchange = exchange_id
                break
            print(f"  {exchange_id} failed or returned no data, trying next ...")

        if df is None or df.empty:
            print(f"\n  ERROR: All exchanges failed for {symbol}.")
            print(f"  Check connectivity and try again.")
            failed_files.append(_cache_filename(symbol, TIMEFRAME))
            continue

        filename = _cache_filename(symbol, TIMEFRAME)
        out_path = Path(filename)

        # Span check BEFORE saving so we don't write a useless file
        if not _validate_span(filename, df):
            failed_files.append(filename)
            continue

        df.to_csv(out_path)   # index="timestamp" UTC, columns=open high low close volume
        saved_files.append(filename)

        n_bars    = len(df)
        date_from = df.index[0]
        date_to   = df.index[-1]
        span_y    = (date_to - date_from).days / 365.25

        print(f"\n  [SAVED]  {filename}")
        print(f"           Exchange  : {used_exchange}")
        print(f"           Bars      : {n_bars:,}")
        print(f"           From      : {date_from.date()}  ({date_from.strftime('%H:%M UTC')})")
        print(f"           To        : {date_to.date()}  ({date_to.strftime('%H:%M UTC')})")
        print(f"           Span      : {span_y:.2f} years  [OK >= {MIN_SPAN_YEARS:.0f}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    if saved_files:
        print("  Files ready for use:")
        for f in saved_files:
            print(f"    {f}")
        print()
        print("  Next steps:")
        print("    1. Download from Colab  (Files panel -> right-click -> Download)")
        print("    2. Drop into  trend_system/data/cache/")
        print("    3. Run:  python main.py   (no --refresh needed)")

        # Auto-download if inside Google Colab
        try:
            from google.colab import files as colab_files
            print("\n  Auto-downloading via google.colab.files ...")
            for f in saved_files:
                colab_files.download(f)
        except ImportError:
            pass

    if failed_files:
        print()
        print("  FAILED or too-short files (do NOT use these):")
        for f in failed_files:
            print(f"    {f}")

    print("=" * 62)

    # Non-zero exit if ANY symbol failed validation or fetch
    if failed_files:
        sys.exit(1)


if __name__ == "__main__":
    main()
