"""
Fetches OHLCV data and caches to CSV.

Crypto (config.INSTRUMENTS[...]["type"] == "crypto"):
  Primary source  : OKX backward-pagination (publicGetMarketHistoryCandles via ccxt).
  Fallback 1      : generic ccxt forward pagination (config.EXCHANGE, e.g. bybit).
  Fallback 2      : Yahoo Finance v8 API via raw requests (SSL verify=False).

Forex (config.INSTRUMENTS[...]["type"] == "forex"):
  Source          : yfinance (forex isn't on crypto exchanges).
  XAUUSD uses "GC=F" (COMEX gold futures) as a free proxy for spot gold.

Returns a clean DataFrame indexed by UTC datetime, columns: open high low close volume.

OKX pagination notes:
  - OKX's 'after' param means "return bars OLDER than this timestamp", so
    forward pagination via ccxt's normal 'since' cursor returns zero bars.
  - We call publicGetMarketHistoryCandles directly (bypasses ccxt's fetch_ohlcv
    remapping) and page BACKWARD from now toward FETCH_ORIGIN in small chunks.
"""
import os
import ssl
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── SSL bypass for TLS-intercepting corporate/ISP proxies ─────────────────────
def _patch_ssl() -> None:
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    os.environ.setdefault("SSL_CERT_FILE", "")
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # type: ignore[attr-defined]
    except Exception:
        pass

_patch_ssl()

# ── Constants ─────────────────────────────────────────────────────────────────
_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

# Oldest date to request when fetching full history
FETCH_ORIGIN = datetime(2018, 1, 1, tzinfo=timezone.utc)

# Conservative per-request bar limits for each exchange
_EXCHANGE_LIMITS = {
    "bybit":   200,
    "binance": 1000,
    "okx":     300,
    "kucoin":  1500,
}
_OKX_HISTORY_LIMIT = 100   # /api/v5/market/history-candles max per request

# Yahoo Finance: map ccxt symbol -> Yahoo ticker (crypto fallback only)
_YF_MAP = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
    "SOL/USDT": "SOL-USD",
    "BNB/USDT": "BNB-USD",
}

_RESAMPLE_RULES = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "_")
    return config.CACHE_DIR / f"{safe}_{timeframe}.csv"


def _resample_to_tf(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule_map = {"1h": "1h", "4h": "4h", "1d": "1D", "1w": "1W"}
    rule = rule_map.get(timeframe, timeframe)
    return df.resample(rule, label="left", closed="left").agg(_RESAMPLE_RULES).dropna()


def _df_from_bars(bars: list) -> pd.DataFrame:
    df = pd.DataFrame(bars, columns=_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def _print_span(symbol: str, df: pd.DataFrame, min_years: Optional[float] = None) -> None:
    """Prints the fetched/loaded date span and WARNs if it's under the minimum."""
    min_years = config.MIN_DATA_YEARS if min_years is None else min_years
    span_days  = (df.index[-1] - df.index[0]).days
    span_years = span_days / 365.25
    print(f"  {symbol}: data spans {df.index[0].date()} to {df.index[-1].date()} "
          f"({span_years:.2f} years, {len(df):,} bars)")
    if span_years < min_years:
        print(f"  WARNING: {symbol} span is only {span_years:.2f} years "
              f"(minimum recommended: {min_years:.0f} years).")
        print(f"  IS/OOS/holdout split and regime breakdown may be statistically unreliable.")


# ── OKX backward-pagination fetch (primary, crypto) ───────────────────────────

def _fetch_okx(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Pages BACKWARD from now toward FETCH_ORIGIN using OKX's history-candles
    endpoint. OKX's 'after' param means "bars older than this timestamp", so
    this is the only pagination direction that actually returns data.
    """
    import ccxt

    floor_ms = int(FETCH_ORIGIN.timestamp() * 1000)
    exchange = ccxt.okx({"enableRateLimit": True, "verify": False})
    exchange.load_markets()

    used_symbol = symbol
    if symbol not in exchange.markets:
        alt = symbol + ":USDT"
        if alt in exchange.markets:
            used_symbol = alt
        else:
            raise RuntimeError(f"{symbol} not available on okx")

    market   = exchange.market(used_symbol)
    inst_id  = market["id"]
    bar_code = exchange.timeframes.get(timeframe)
    if bar_code is None:
        raise ValueError(f"Timeframe '{timeframe}' not in OKX timeframes map")

    sleep_s  = max(exchange.rateLimit / 1000, 0.12)
    all_bars: list = []
    after_cursor = None

    print(f"  [okx-backward] Fetching {inst_id} {bar_code} back to {FETCH_ORIGIN.date()} ...", flush=True)

    while True:
        req = {"instId": inst_id, "bar": bar_code, "limit": str(_OKX_HISTORY_LIMIT)}
        if after_cursor is not None:
            req["after"] = str(after_cursor)

        try:
            response = exchange.publicGetMarketHistoryCandles(req)
        except AttributeError:
            response = exchange.public_get_market_history_candles(req)

        rows = response.get("data", [])
        if not rows:
            break

        oldest_ts_in_batch = int(rows[-1][0])   # OKX returns newest-first

        for row in rows:
            ts = int(row[0])
            if ts < floor_ms:
                continue
            try:
                all_bars.append([ts, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
            except (ValueError, IndexError):
                continue

        if len(all_bars) % 2000 < _OKX_HISTORY_LIMIT:
            print(f"    ... {len(all_bars):,} bars", flush=True)

        after_cursor = oldest_ts_in_batch
        if oldest_ts_in_batch <= floor_ms or len(rows) < _OKX_HISTORY_LIMIT:
            break

        time.sleep(sleep_s)

    if not all_bars:
        raise RuntimeError(f"OKX returned no usable bars for {symbol}")

    df = _df_from_bars(all_bars)
    df = df[~df.index.duplicated(keep="last")]
    print(f"    Done: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})", flush=True)
    return df


# ── ccxt fetch (fallback #1, crypto) ──────────────────────────────────────────

def _fetch_ccxt(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Fetches OHLCV from config.EXCHANGE via ccxt using forward pagination.
    Starts from FETCH_ORIGIN and paginates forward until present.
    """
    import ccxt

    limit    = _EXCHANGE_LIMITS.get(config.EXCHANGE, 500)
    exchange = getattr(ccxt, config.EXCHANGE)({"enableRateLimit": True, "verify": False})
    since_ms = int(FETCH_ORIGIN.timestamp() * 1000)

    all_bars: list = []
    print(f"  [ccxt/{config.EXCHANGE}] Fetching {symbol} {timeframe} "
          f"from {FETCH_ORIGIN.date()} to present ...", flush=True)

    while True:
        bars = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=limit
        )

        if not bars:
            break

        all_bars.extend(bars)
        new_since = bars[-1][0] + 1

        # Reached the end of available history
        if len(bars) < limit:
            break

        # Safety: no forward progress (shouldn't happen but guards infinite loops)
        if new_since <= since_ms:
            break

        since_ms = new_since

        if len(all_bars) % 2000 < limit:
            print(f"    ... {len(all_bars):,} bars", flush=True)

        time.sleep(max(exchange.rateLimit / 1000, 0.12))

    if not all_bars:
        raise RuntimeError(f"ccxt returned no OHLCV bars for {symbol}")

    df = _df_from_bars(all_bars)
    df = df[~df.index.duplicated(keep="last")]
    print(f"    Done: {len(df):,} bars  "
          f"({df.index[0].date()} to {df.index[-1].date()})", flush=True)
    return df


# ── Yahoo Finance fetch (fallback #2, crypto) ─────────────────────────────────

def _fetch_yahoo(symbol: str, timeframe: str, since_days: int) -> pd.DataFrame:
    """
    Falls back to Yahoo Finance v8 chart API via raw requests (SSL verify=False).
    Yahoo only keeps 1H data for ~720 days; chunks in 60-day windows.
    """
    import requests

    ticker = _YF_MAP.get(symbol)
    if ticker is None:
        raise ValueError(f"No Yahoo mapping for {symbol}. Add it to _YF_MAP in loader.py.")

    session = requests.Session()
    session.verify = False
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})

    yf_days     = min(since_days, 720)
    now_ts      = int(datetime.now(timezone.utc).timestamp())
    start_ts    = int((datetime.now(timezone.utc) - timedelta(days=yf_days)).timestamp())
    chunk_secs  = 60 * 24 * 3600    # 60-day windows

    print(f"  [yahoo-api] Fetching {ticker} 1h -> resample to {timeframe} ...", flush=True)

    all_frames: list = []
    chunk_start = start_ts

    while chunk_start < now_ts:
        chunk_end = min(chunk_start + chunk_secs, now_ts)
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={chunk_start}&period2={chunk_end}&interval=1h&includePrePost=false"
        )
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            result = data["chart"]["result"]
            if result:
                r  = result[0]
                ts = r["timestamp"]
                q  = r["indicators"]["quote"][0]
                adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
                frame = pd.DataFrame(
                    {"open": q["open"], "high": q["high"], "low": q["low"],
                     "close": adj, "volume": q["volume"]},
                    index=pd.to_datetime(ts, unit="s", utc=True),
                )
                all_frames.append(frame)
        except Exception as e:
            print(f"    chunk failed: {e}", flush=True)

        chunk_start = chunk_end
        time.sleep(0.4)

    if not all_frames:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker}")

    raw = pd.concat(all_frames).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")].dropna()
    print(f"    Done: {len(raw):,} 1h bars  "
          f"({raw.index[0].date()} to {raw.index[-1].date()})", flush=True)

    if timeframe != "1h":
        raw = _resample_to_tf(raw, timeframe)
        print(f"    Resampled to {timeframe}: {len(raw):,} bars", flush=True)
    return raw


# ── yfinance fetch (primary, forex) ───────────────────────────────────────────

def _fetch_yfinance(ticker: str, timeframe: str) -> pd.DataFrame:
    """
    Fetches OHLCV via yfinance and resamples to the requested timeframe.
    yfinance only supports "1h"/"1d"-style native intervals, so intraday
    timeframes are fetched as 1h and resampled (e.g. 4h from 1h).
    yfinance also caps intraday (1h) history at ~730 days.
    """
    import yfinance as yf

    intraday_tfs   = {"1h", "4h"}
    fetch_interval = "1h" if timeframe in intraday_tfs else "1d"
    period         = "730d" if fetch_interval == "1h" else "max"

    print(f"  [yfinance] Fetching {ticker} {fetch_interval} (period={period}) "
          f"-> resample to {timeframe} ...", flush=True)

    # yfinance's HTTP client (curl_cffi) doesn't honour the module-level SSL
    # monkeypatch above, so TLS-intercepting proxies need verify=False passed
    # explicitly via a custom session.
    session = None
    try:
        import curl_cffi.requests as ccr
        session = ccr.Session(verify=False, impersonate="chrome")
    except ImportError:
        pass

    raw = yf.Ticker(ticker, session=session).history(period=period, interval=fetch_interval, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    raw.index = pd.to_datetime(raw.index, utc=True)
    raw.index.name = "timestamp"
    raw = raw.rename(columns=str.lower)
    raw = raw[["open", "high", "low", "close", "volume"]]
    raw = raw[~raw.index.duplicated(keep="last")].sort_index().dropna(subset=["open", "high", "low", "close"])
    raw["volume"] = raw["volume"].fillna(0.0)

    print(f"    Done: {len(raw):,} {fetch_interval} bars "
          f"({raw.index[0].date()} to {raw.index[-1].date()})", flush=True)

    if timeframe != fetch_interval:
        raw = _resample_to_tf(raw, timeframe)
        print(f"    Resampled to {timeframe}: {len(raw):,} bars", flush=True)
    return raw


# ── Staleness refresh ─────────────────────────────────────────────────────────

def _extend_cache(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """Append any bars newer than the cache tail using ccxt (or Yahoo as fallback)."""
    last_ts  = df.index[-1]
    since_ms = int(last_ts.timestamp() * 1000) + 1
    limit    = _EXCHANGE_LIMITS.get(config.EXCHANGE, 500)

    try:
        import ccxt
        exchange = getattr(ccxt, config.EXCHANGE)({"enableRateLimit": True, "verify": False})
        bars = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=limit
        )
        if bars:
            new_df = _df_from_bars(bars)
            df = pd.concat([df, new_df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
    except Exception as e:
        print(f"  Cache extend failed ({e}), using existing data.")
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load(
    symbol:        str,
    timeframe:     str  = config.TIMEFRAME,
    since_days:    int  = config.SINCE_DAYS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return clean OHLCV DataFrame. Uses CSV cache when available and fresh."""
    path = _cache_path(symbol, timeframe)
    instrument_cfg  = config.INSTRUMENTS.get(symbol, {})
    instrument_type = instrument_cfg.get("type", "crypto")

    if path.exists() and not force_refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)

        tf_hours = {"1h": 1, "4h": 4, "1d": 24}.get(timeframe, 4)
        if datetime.now(timezone.utc) - df.index[-1] > timedelta(hours=tf_hours * 2):
            print(f"  Cache stale for {symbol} - extending ...", flush=True)
            if instrument_type == "forex":
                try:
                    df = _fetch_yfinance(instrument_cfg["yf_ticker"], timeframe)
                except Exception as e:
                    print(f"  Refresh failed ({e}), using existing cache.")
            else:
                df = _extend_cache(df, symbol, timeframe)
            df.to_csv(path)

        print(f"  Loaded {symbol} {timeframe}: {len(df):,} bars  "
              f"({df.index[0].date()} to {df.index[-1].date()}) from cache.")
        _print_span(symbol, df)
        return df

    # Full fetch
    if instrument_type == "forex":
        ticker = instrument_cfg.get("yf_ticker")
        if not ticker:
            raise ValueError(f"No yfinance ticker configured for {symbol}. Add it to config.INSTRUMENTS.")
        df = _fetch_yfinance(ticker, timeframe)
    else:
        df = None
        try:
            df = _fetch_okx(symbol, timeframe)
        except Exception as e:
            print(f"  OKX backward fetch failed ({type(e).__name__}: {e})")
            print(f"  Falling back to ccxt/{config.EXCHANGE} ...")
            try:
                df = _fetch_ccxt(symbol, timeframe)
            except Exception as e2:
                print(f"  ccxt failed ({type(e2).__name__}: {e2})")
                print(f"  Falling back to Yahoo Finance ...")
                df = _fetch_yahoo(symbol, timeframe, since_days)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"  Cached {len(df):,} bars -> {path.name}")
    _print_span(symbol, df)
    return df
