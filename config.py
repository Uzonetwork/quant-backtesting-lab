from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
CACHE_DIR  = ROOT_DIR / "data" / "cache"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_LOG_PATH = ROOT_DIR / "results_log.csv"   # every run appends a row here (anti-data-mining)

# ── Instruments ──────────────────────────────────────────────────────────────────
# type "crypto" -> fetched via OKX backward-pagination (ccxt/bybit + Yahoo as fallbacks).
# type "forex"  -> fetched via yfinance (forex isn't on crypto exchanges).
# yf_ticker is the Yahoo Finance ticker: used as fallback for crypto, primary for forex.
INSTRUMENTS = {
    "BTC/USDT": {"type": "crypto", "exchange": "okx", "yf_ticker": "BTC-USD"},
    "ETH/USDT": {"type": "crypto", "exchange": "okx", "yf_ticker": "ETH-USD"},
    "XAUUSD":   {"type": "forex",  "yf_ticker": "GC=F"},        # COMEX gold futures, proxy for spot XAUUSD
    "GBPUSD":   {"type": "forex",  "yf_ticker": "GBPUSD=X"},
}

SYMBOLS    = [s for s, c in INSTRUMENTS.items() if c["type"] == "crypto"]   # back-compat for export_setups.py
EXCHANGE   = "bybit"       # generic ccxt fallback exchange for crypto (OKX backward-pagination is primary)
TIMEFRAME  = "4h"
SINCE_DAYS = 1825          # used only by the Yahoo Finance fallback; primary fetchers go to earliest available
MIN_DATA_YEARS = 2.0       # WARN (not fatal) if fetched history span is shorter than this

# ── Strategy selection ────────────────────────────────────────────────────────
DEFAULT_STRATEGY = "trend"   # trend | meanrev | breakout  (see strategies/)

# ── Trend-following strategy (Donchian breakout + chandelier exit) ────────────
DONCHIAN_PERIOD = 50       # breakout lookback (bars)
SMA_PERIOD      = 200      # direction filter
ATR_PERIOD      = 14
ATR_STOP_MULT   = 2.5      # initial stop = entry ± ATR_STOP_MULT * ATR
ATR_TRAIL_MULT  = 3.0      # chandelier trailing-stop multiplier (engine applies this to ALL strategies)

# ── Mean-reversion strategy (Bollinger/RSI fade) ──────────────────────────────
MEANREV_BB_PERIOD      = 20
MEANREV_BB_STD         = 2.0
MEANREV_RSI_PERIOD     = 14
MEANREV_RSI_OVERSOLD   = 30
MEANREV_RSI_OVERBOUGHT = 70
MEANREV_ATR_PERIOD     = 14
MEANREV_ATR_STOP_MULT  = 2.0

# ── Breakout strategy (volatility-squeeze breakout) ───────────────────────────
BREAKOUT_BB_PERIOD     = 20
BREAKOUT_BB_MULT       = 2.0
BREAKOUT_KC_PERIOD     = 20
BREAKOUT_KC_MULT       = 1.5
BREAKOUT_RANGE_PERIOD  = 20
BREAKOUT_ATR_PERIOD    = 14
BREAKOUT_ATR_STOP_MULT = 2.0

# ── Sweep-and-reclaim strategy (liquidity-sweep reversal) ─────────────────────
SWEEP_ATR_PERIOD           = 14
SWEEP_ATR_STOP_BUFFER_MULT = 0.25   # small buffer beyond the sweep candle's extreme

# ── Risk / sizing ──────────────────────────────────────────────────────────────
INITIAL_EQUITY  = 100_000.0
RISK_FRACTION   = 0.0075   # 0.75 % of equity per trade
MAX_POSITION_PCT = 0.20    # hard cap: no single position > 20 % of equity

# Kill-switch (plumbing only — set None to disable)
MAX_CONSECUTIVE_LOSSES = None   # e.g. 6
MAX_TRADES_PER_DAY     = None   # e.g. 3

# ── Execution costs ────────────────────────────────────────────────────────────
COMMISSION_PCT = 0.0006    # 0.06 % per side (maker/taker blended)
SLIPPAGE_PCT   = 0.0005    # 0.05 % per side

# ── Train / test / holdout split ────────────────────────────────────────────────
# Chronological, non-shuffled split. FINAL_HOLDOUT is never evaluated during
# development -- only with the explicit --final CLI flag, and only once you
# mean it. See README "Anti-Data-Mining Guards".
IS_RATIO      = 0.60   # in-sample      (development)
OOS_RATIO     = 0.20   # out-of-sample  (development, but not fit to)
HOLDOUT_RATIO = 0.20   # final holdout  (locked; --final only; one-shot)
