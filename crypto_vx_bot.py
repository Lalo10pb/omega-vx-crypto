import ccxt
import csv
import json
import os
import sys
import time

# Force unbuffered output for Render logs
sys.stdout.reconfigure(line_buffering=True)
from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import gspread
import numpy as np
import pandas as pd
import requests
from ccxt.base.errors import NetworkError, ExchangeError
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials

env_path = os.path.join(os.path.dirname(__file__), ".env")
print(f"📦 Loading environment from: {env_path}")
if load_dotenv(dotenv_path=env_path):
    print("🔑 Environment variables loaded from .env.")
else:
    print("⚠️ .env file not found or could not be loaded; relying on existing environment.")


@dataclass
class BotConfig:
    LOOP_SLEEP_SECONDS: int = int(os.getenv("LOOP_SLEEP_SECONDS", 300))
    POSITION_CHECK_INTERVAL: int = int(os.getenv("POSITION_CHECK_INTERVAL", 60))
    ALLOW_MARKET_FALLBACK: bool = os.getenv("ALLOW_MARKET_FALLBACK", "true").lower() == "true"
    DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"
    ALLOW_TRADING: bool = os.getenv("ALLOW_TRADING", "true").lower() == "true"
    BOT_LOG_LEVEL: str = os.getenv("BOT_LOG_LEVEL", "info")
    BOT_LOG_JSON: bool = os.getenv("BOT_LOG_JSON", "false").lower() == "true"


@dataclass
class RegimeState:
    score: float
    status: str
    reason: str
    can_trade: bool
    risk_scaler: float


config = BotConfig()


def log_event(level: str, source: str, message: str) -> None:
    """Centralized event logger (console + Telegram)."""
    entry = f"[{level.upper()}] [{source}] {message}"
    print(entry)
    if "error" in level.lower() or "warn" in level.lower():
        send_telegram_alert(entry)


print("🟢 OMEGA-VX-CRYPTO bot started.")
open_positions = set()
open_positions_data = {}
last_buy_time = defaultdict(lambda: 0)
last_trade_time = defaultdict(lambda: 0)
symbol_cooldowns = defaultdict(lambda: 0.0)
COOLDOWN_SECONDS = 30 * 60  # 30 minutes
TRADE_COOLDOWN_SECONDS = 60 * 60  # 1 hour global cooldown per symbol

markets_cache: Dict[str, dict] = {}
markets_last_load: float = 0.0
ohlcv_cache: Dict[Tuple[str, str, int], Dict[str, object]] = {}

regime_last_check: float = 0.0
# Initialize with a safe default state
regime_last_result: RegimeState = RegimeState(
    score=50.0, 
    status="INIT", 
    reason="Initializing...", 
    can_trade=True, 
    risk_scaler=0.5
)
last_regime_warning_ts: float = 0.0
last_regime_warning_reason: str = ""

orderbook_cache: Dict[Tuple[str, float], Dict[str, object]] = {}
listing_age_cache: Dict[str, Dict[str, float]] = {}

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", 5))
MAX_TOTAL_EXPOSURE_USD = float(os.getenv("MAX_TOTAL_EXPOSURE_USD", 500.0))
STATE_PATH = os.getenv("BOT_STATE_PATH", "bot_state.json")
LIMIT_PRICE_BUFFER = float(os.getenv("LIMIT_PRICE_BUFFER", 0.001))
MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME_24H", 50_000))
_MIN_QUOTE_VOLUME_FALLBACK = float(
    os.getenv("MIN_QUOTE_VOLUME_FALLBACK", 30_000)
)
MIN_QUOTE_VOLUME_FALLBACK = max(0.0, min(_MIN_QUOTE_VOLUME_FALLBACK, MIN_QUOTE_VOLUME_24H))
MAX_SPREAD_PERCENT = float(os.getenv("MAX_SPREAD_PERCENT", 0.35))  # expressed in percent
MAX_ATR_PERCENT = float(os.getenv("MAX_ATR_PERCENT", 0.07))
FRESH_SIGNAL_LOOKBACK = int(os.getenv("FRESH_SIGNAL_LOOKBACK", 3))
SCANNER_LOG_PATH = os.getenv("SCANNER_LOG_PATH", "scanner_evaluation_log.csv")
SCANNER_SNAPSHOT_PATH = os.getenv("SCANNER_SNAPSHOT_PATH", "scanner_snapshot_log.csv")
SCANNER_MAX_CANDIDATES = max(int(os.getenv("SCANNER_MAX_CANDIDATES", 1)), 1)
last_telegram_notify = datetime.now(timezone.utc) - timedelta(hours=12)  # Avoid spamming alerts
last_loop_start_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
last_regime_metrics: Dict[str, object] = {}
last_weekly_summary_date: Optional[str] = None
last_portfolio_snapshot: float = 0.0
last_health_heartbeat: float = 0.0
MARKET_REFRESH_SECONDS = int(os.getenv("MARKET_REFRESH_SECONDS", 900))
OHLCV_CACHE_TTL_SECONDS = int(os.getenv("OHLCV_CACHE_TTL_SECONDS", 90))
MULTI_TF_CACHE_TTL_SECONDS = int(os.getenv("MULTI_TF_CACHE_TTL_SECONDS", 300))
PREFERRED_QUOTES = [
    q.strip().upper()
    for q in os.getenv("PREFERRED_QUOTES", "USD,USDT,USDC,USDD,DAI,BUSD").split(",")
    if q.strip()
]
HARD_STOP_PERCENT = float(os.getenv("HARD_STOP_PERCENT", 3.0))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", 0.02))
ATR_TRAIL_MULTIPLIER = float(os.getenv("ATR_TRAIL_MULTIPLIER", 2.0))
ATR_TAKE_PROFIT_MULTIPLIER = float(os.getenv("ATR_TAKE_PROFIT_MULTIPLIER", 3.5))
ATR_HARD_STOP_MULTIPLIER = float(os.getenv("ATR_HARD_STOP_MULTIPLIER", 3.0))
REGIME_SYMBOL = os.getenv("REGIME_SYMBOL", "BTC/USD")
REGIME_TIMEFRAME = os.getenv("REGIME_TIMEFRAME", "4h")
REGIME_EMA_FAST = int(os.getenv("REGIME_EMA_FAST", 21))
REGIME_EMA_SLOW = int(os.getenv("REGIME_EMA_SLOW", 50))
REGIME_LOOKBACK = int(os.getenv("REGIME_LOOKBACK", 180))
REGIME_MIN_ROC = float(os.getenv("REGIME_MIN_ROC", 0.2))
REGIME_CACHE_SECONDS = int(os.getenv("REGIME_CACHE_SECONDS", 300))
REGIME_ROC_GRACE = max(float(os.getenv("REGIME_ROC_GRACE", 0.5)), 0.0)
REGIME_EMA_GRACE_BARS = max(int(os.getenv("REGIME_EMA_GRACE_BARS", 3)), 1)
REGIME_WARN_COOLDOWN_SECONDS = max(int(os.getenv("REGIME_WARN_COOLDOWN_SECONDS", 3600)), 0)
REGIME_METRICS_LOG_PATH = os.getenv("REGIME_METRICS_LOG_PATH", "regime_metrics_log.csv")
ORDERBOOK_CACHE_TTL_SECONDS = int(os.getenv("ORDERBOOK_CACHE_TTL_SECONDS", 30))
DEPTH_ORDERBOOK_LIMIT = int(os.getenv("DEPTH_ORDERBOOK_LIMIT", 50))
DEPTH_BAND_PERCENT = float(os.getenv("DEPTH_BAND_PERCENT", 0.5))
MIN_DEPTH_IMBALANCE = float(os.getenv("MIN_DEPTH_IMBALANCE", 1.1))
MIN_DEPTH_NOTIONAL_USD = float(os.getenv("MIN_DEPTH_NOTIONAL_USD", 1500.0))
LISTING_AGE_LOOKBACK_DAYS = int(os.getenv("LISTING_AGE_LOOKBACK_DAYS", 365))
LISTING_AGE_CACHE_SECONDS = int(os.getenv("LISTING_AGE_CACHE_SECONDS", 3600))
NEW_LISTING_BOOST_DAYS = int(os.getenv("NEW_LISTING_BOOST_DAYS", 30))
RESTRICTED_SYMBOLS = {
    s.strip().upper()
    for s in os.getenv("RESTRICTED_SYMBOLS", "EUR/USD,EUR/USDT,EUR/USDC,GBP/USD,AUD/USD,CAD/USD,JPY/USD").split(",")
    if s.strip()
}
# Nebraska/Regional restricted symbols (will be populated dynamically + defaults)
RUNTIME_RESTRICTED_SYMBOLS: Set[str] = set()
BLUE_CHIP_SYMBOLS = {
    s.strip().upper()
    for s in os.getenv("BLUE_CHIP_SYMBOLS", "BTC,ETH,SOL,BNB,XRP,DOGE,ADA").split(",")
    if s.strip()
}
PORTFOLIO_SNAPSHOT_INTERVAL_SECONDS = int(os.getenv("PORTFOLIO_SNAPSHOT_INTERVAL_SECONDS", 1800))
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "google_credentials.json")
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", 10800))
SKIP_BOT_INIT = os.getenv("OMEGA_SKIP_INIT", "false").lower() == "true"

SOFT_VOLUME_MULTIPLIER = float(os.getenv("SOFT_VOLUME_MULTIPLIER", 0.6))
SOFT_SPREAD_BUFFER = float(os.getenv("SOFT_SPREAD_BUFFER", 0.15))
SOFT_DEPTH_BUFFER = float(os.getenv("SOFT_DEPTH_BUFFER", 0.05))
MIN_CANDIDATE_SCORE = float(os.getenv("MIN_CANDIDATE_SCORE", 4.0))
SCANNER_NEAR_MISS_PATH = os.getenv("SCANNER_NEAR_MISS_PATH", "scanner_near_miss_log.csv")

SCORING_WEIGHTS = {
    "ema_stack": float(os.getenv("WEIGHT_EMA_STACK", 2.5)),
    "fresh_cross": float(os.getenv("WEIGHT_FRESH_CROSS", 1.5)),
    "rsi": float(os.getenv("WEIGHT_RSI", 1.2)),
    "rsi_penalty": float(os.getenv("WEIGHT_RSI_PENALTY", 0.6)),
    "ema_slope": float(os.getenv("WEIGHT_EMA_SLOPE", 1.4)),
    "volume_ratio": float(os.getenv("WEIGHT_VOLUME_RATIO", 1.0)),
    "adx": float(os.getenv("WEIGHT_ADX", 1.0)),
    "macd": float(os.getenv("WEIGHT_MACD", 1.0)),
    "roc": float(os.getenv("WEIGHT_ROC", 0.9)),
    "timeframe_alignment": float(os.getenv("WEIGHT_TIMEFRAME_ALIGNMENT", 1.1)),
    "depth": float(os.getenv("WEIGHT_DEPTH", 0.8)),
    "new_listing": float(os.getenv("WEIGHT_NEW_LISTING", 0.5)),
    "volatility": float(os.getenv("WEIGHT_VOLATILITY", 0.8)),
    "regime_bull": float(os.getenv("WEIGHT_REGIME_BULL", 0.7)),
    "regime_bear": float(os.getenv("WEIGHT_REGIME_BEAR", 1.2)),
    "volume_penalty": float(os.getenv("WEIGHT_VOLUME_PENALTY", 2.0)),
    "spread_penalty": float(os.getenv("WEIGHT_SPREAD_PENALTY", 1.5)),
    "depth_penalty": float(os.getenv("WEIGHT_DEPTH_PENALTY", 1.3)),
}

 # === CONFIGURATION ===
# trade_amount_usd = 25  # USD amount per trade (adjusted to increase capital usage)

# === Paths ===
TRADE_LOG_PATH = "crypto_trade_log.csv"
TRADE_LOG_PATH_BACKUP = "crypto_trade_log_backup.csv"
PORTFOLIO_LOG_PATH = "crypto_portfolio_log.csv"
TRADE_FEATURE_LOG_PATH = os.getenv("TRADE_FEATURE_LOG_PATH", "trade_feature_log.csv")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")
EQUITY_LOG_PATH = os.path.join(LOGS_DIR, "equity_log.csv")


def ensure_directory(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


ensure_directory(LOGS_DIR)



def validate_startup_config() -> None:
    """Validate critical configuration to avoid running half-configured."""
    errors: List[str] = []

    if config.LOOP_SLEEP_SECONDS <= 0:
        errors.append("LOOP_SLEEP_SECONDS must be > 0.")
    if config.POSITION_CHECK_INTERVAL <= 0:
        errors.append("POSITION_CHECK_INTERVAL must be > 0.")
    if MAX_OPEN_POSITIONS <= 0:
        errors.append("MAX_OPEN_POSITIONS must be > 0.")
    if MAX_TOTAL_EXPOSURE_USD <= 0:
        errors.append("MAX_TOTAL_EXPOSURE_USD must be > 0.")
    if not (0 < RISK_PER_TRADE_PCT <= 1):
        errors.append("RISK_PER_TRADE_PCT must be between 0 and 1 (fraction of equity).")
    if HEARTBEAT_INTERVAL_SECONDS < 0:
        errors.append("HEARTBEAT_INTERVAL_SECONDS cannot be negative.")

    credentials_present = GOOGLE_CREDENTIALS_PATH and os.path.exists(GOOGLE_CREDENTIALS_PATH)
    google_sheet_id = os.getenv("GOOGLE_SHEET_ID")
    portfolio_sheet_name = os.getenv("PORTFOLIO_SHEET_NAME")
    if credentials_present:
        if not google_sheet_id:
            errors.append("GOOGLE_SHEET_ID is required when GOOGLE_CREDENTIALS_PATH is present.")
        if not portfolio_sheet_name:
            errors.append("PORTFOLIO_SHEET_NAME is required when GOOGLE_CREDENTIALS_PATH is present.")

    if errors:
        for err in errors:
            log_event("error", "Config", err)
        raise SystemExit("Startup validation failed; fix configuration and restart.")



if not SKIP_BOT_INIT:
    validate_startup_config()


def call_with_retries(func, *args, attempts: int = 3, backoff: float = 1.5, base_delay: float = 0.5, **kwargs):
    """
    Small retry helper for exchange/network calls with exponential backoff.
    Retries NetworkError/ExchangeError and RequestException.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return func(*args, **kwargs)
        except (NetworkError, ExchangeError, requests.exceptions.RequestException) as err:
            last_err = err
            if attempt >= attempts:
                break
            sleep_s = base_delay * (backoff ** (attempt - 1))
            log_event("warn", "Retry", f"{func.__name__} attempt {attempt}/{attempts} failed: {err}; retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)
        except Exception as err:  # Non-retriable
            last_err = err
            break
    if last_err:
        raise last_err
    return None


def log_regime_metrics_entry(
    timestamp: datetime,
    ema_fast: float,
    ema_slow: float,
    roc_value: float,
    bullish: bool,
    reason: str,
    grace_active: bool,
) -> None:
    if not REGIME_METRICS_LOG_PATH:
        return
    log_dir = os.path.dirname(REGIME_METRICS_LOG_PATH)
    if log_dir:
        ensure_directory(log_dir)
    try:
        file_exists = os.path.exists(REGIME_METRICS_LOG_PATH)
        with open(REGIME_METRICS_LOG_PATH, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "ema_fast",
                    "ema_slow",
                    "roc_pct",
                    "bullish",
                    "grace_active",
                    "reason",
                ])
            writer.writerow([
                timestamp.isoformat(),
                f"{ema_fast:.6f}",
                f"{ema_slow:.6f}",
                f"{roc_value:.6f}",
                "1" if bullish else "0",
                "1" if grace_active else "0",
                reason,
            ])
    except Exception as err:
        print(f"⚠️ Failed to log regime metrics: {err}")

# === Helper Functions for EMA and RSI ===
def calculate_ema(prices, period=20):
    prices = np.array(prices, dtype=float)
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    return np.convolve(prices, weights, mode='valid')

def calculate_rsi(prices, period=14):
    prices = np.array(prices, dtype=float)
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0:
        first_rsi = 100.0 if up > 0 else 50.0
    elif up == 0:
        first_rsi = 0.0
    else:
        rs = up / down
        first_rsi = 100. - 100. / (1. + rs)
    rsi = [first_rsi]

    for delta in deltas[period:]:
        up_val = max(delta, 0)
        down_val = -min(delta, 0)
        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period
        if down == 0:
            rsi.append(100.0 if up > 0 else 50.0)
        elif up == 0:
            rsi.append(0.0)
        else:
            rs = up / down
            rsi.append(100. - 100. / (1. + rs))
    return rsi


def calculate_ema_series(prices: List[float], period: int) -> List[float]:
    if len(prices) < period:
        return []
    alpha = 2 / (period + 1)
    ema_values: List[float] = []
    ema: Optional[float] = None
    for price in prices:
        if ema is None:
            ema = price
        else:
            ema = (price - ema) * alpha + ema
        ema_values.append(ema)
    return ema_values


def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    size = min(len(highs), len(lows), len(closes))
    if size < period + 1:
        return []
    highs = np.array(highs[-(size):], dtype=float)
    lows = np.array(lows[-(size):], dtype=float)
    closes = np.array(closes[-(size):], dtype=float)

    plus_dm = np.zeros(size)
    minus_dm = np.zeros(size)
    tr = np.zeros(size)

    for i in range(1, size):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        tr_components = [highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])]
        tr[i] = max(tr_components)

    def _smooth(values: np.ndarray) -> np.ndarray:
        smoothed = np.zeros_like(values)
        smoothed[period] = values[1:period + 1].sum()
        for i in range(period + 1, len(values)):
            smoothed[i] = smoothed[i - 1] - (smoothed[i - 1] / period) + values[i]
        return smoothed

    smoothed_tr = _smooth(tr)
    smoothed_plus = _smooth(plus_dm)
    smoothed_minus = _smooth(minus_dm)

    plus_di = np.zeros(size)
    minus_di = np.zeros(size)
    np.divide(smoothed_plus, smoothed_tr, out=plus_di, where=smoothed_tr > 0)
    np.divide(smoothed_minus, smoothed_tr, out=minus_di, where=smoothed_tr > 0)
    plus_di *= 100
    minus_di *= 100

    direction_sum = plus_di + minus_di
    dx = np.zeros(size)
    np.divide(
        np.abs(plus_di - minus_di),
        direction_sum,
        out=dx,
        where=direction_sum > 0
    )
    dx *= 100

    adx = np.zeros(size)
    adx[:period * 2] = dx[:period * 2]
    for i in range(period * 2, size):
        adx[i] = ((adx[i - 1] * (period - 1)) + dx[i]) / period

    return adx.tolist()


def compute_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
    if len(prices) < slow + signal:
        return [], [], []
    ema_fast = calculate_ema_series(prices, fast)
    ema_slow = calculate_ema_series(prices, slow)
    if not ema_fast or not ema_slow:
        return [], [], []
    ema_fast = np.array(ema_fast[-len(ema_slow):])
    ema_slow = np.array(ema_slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema_series(macd_line.tolist(), signal)
    if not signal_line:
        return macd_line.tolist(), [], []
    signal_line = np.array(signal_line[-len(macd_line):])
    histogram = macd_line - signal_line
    return macd_line.tolist(), signal_line.tolist(), histogram.tolist()


def rate_of_change(prices: List[float], period: int = 10) -> Optional[float]:
    if len(prices) <= period:
        return None
    past_price = prices[-period - 1]
    latest_price = prices[-1]
    if past_price <= 0:
        return None
    return ((latest_price - past_price) / past_price) * 100


def compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if min(len(highs), len(lows), len(closes)) < period + 1:
        return None
    true_ranges = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return None
    atr_window = true_ranges[-period:]
    atr_values = calculate_ema_series(atr_window, period)
    if not atr_values:
        return None
    return float(atr_values[-1])


def send_telegram_alert(message):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("⚠️ Telegram credentials missing.")
        return
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ Telegram alert failed: {response.text}")
    except Exception as e:
        print(f"❌ Telegram exception: {str(e)}")

# === Exchange Client Loader ===
exchange_name = os.getenv("EXCHANGE", "okx").lower()


def validate_exchange_credentials(name: str) -> None:
    required_vars = []
    if name == "bybit":
        required_vars = ["BYBIT_API_KEY", "BYBIT_API_SECRET"]
    elif name == "kraken":
        required_vars = ["KRAKEN_API_KEY", "KRAKEN_API_SECRET"]
    else:
        required_vars = ["OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE"]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing required credentials for {name.upper()}: {', '.join(missing)}")
        exit(1)


exchange = None
if not SKIP_BOT_INIT:
    validate_exchange_credentials(exchange_name)

    if exchange_name == "bybit":
        exchange = ccxt.bybit({
            'apiKey': os.getenv("BYBIT_API_KEY"),
            'secret': os.getenv("BYBIT_API_SECRET"),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        if os.getenv("BYBIT_API_TESTNET", "false").lower() == "true":
            exchange.set_sandbox_mode(True)
        print("🪙 BYBIT TESTNET mode active.")
        send_telegram_alert("🪙 BYBIT TESTNET mode active.")
    elif exchange_name == "kraken":
        exchange = ccxt.kraken({
            'apiKey': os.getenv("KRAKEN_API_KEY"),
            'secret': os.getenv("KRAKEN_API_SECRET"),
            'enableRateLimit': True,
        })
        print("🪙 KRAKEN LIVE mode active.")
        send_telegram_alert("🪙 KRAKEN LIVE mode active.")
    else:
        exchange = ccxt.okx({
            'apiKey': os.getenv("OKX_API_KEY"),
            'secret': os.getenv("OKX_API_SECRET"),
            'password': os.getenv("OKX_API_PASSPHRASE"),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        # Set sandbox mode according to OKX_API_TESTNET env variable
        if os.getenv("OKX_API_TESTNET", "false").lower() == "true":
            exchange.set_sandbox_mode(True)
            print("🪙 OKX TESTNET mode active.")
            send_telegram_alert("🪙 OKX TESTNET mode active.")
        else:
            exchange.set_sandbox_mode(False)
            print("🪙 OKX LIVE MODE active.")
            send_telegram_alert("🪙 OKX LIVE MODE active.")
else:
    print("⏸️ OMEGA_SKIP_INIT set; skipping exchange initialization.")


# === Order Utility Helpers ===
DEFAULT_QUOTE_ASSET = os.getenv("DEFAULT_QUOTE_ASSET")
if DEFAULT_QUOTE_ASSET:
    DEFAULT_QUOTE_ASSET = DEFAULT_QUOTE_ASSET.upper()
else:
    DEFAULT_QUOTE_ASSET = "USD" if exchange_name == "kraken" else "USDT"

def extract_valid_price(ticker: dict) -> Optional[float]:
    """Return the first positive price from common ticker fields."""
    if not isinstance(ticker, dict):
        return None
    candidates = [
        ticker.get('last'),
        ticker.get('close'),
        ticker.get('ask'),
        ticker.get('bid'),
        ticker.get('average'),
    ]
    for candidate in candidates:
        if isinstance(candidate, (int, float)) and candidate > 0:
            return float(candidate)
    return None


def wait_for_order_fill(symbol: str, order: dict, timeout: int = 45, poll_interval: int = 3) -> dict:
    """Poll the exchange until the order is filled, cancelled, or times out."""
    order_id = order.get('id')
    if not order_id:
        return order

    deadline = time.time() + timeout
    latest = order
    while time.time() < deadline:
        try:
            latest = exchange.fetch_order(order_id, symbol)
        except Exception as fetch_err:
            print(f"⚠️ Failed to fetch order {order_id}: {fetch_err}")
            time.sleep(poll_interval)
            continue

        status = latest.get('status')
        filled = latest.get('filled') or 0
        amount = latest.get('amount') or 0
        if status == 'closed' or (filled and filled >= amount):
            return latest
        if status in {'canceled', 'rejected', 'expired'}:
            return latest
        time.sleep(poll_interval)

    print(f"⏰ Order {order_id} timeout reached; attempting to cancel.")
    try:
        exchange.cancel_order(order_id, symbol)
    except Exception as cancel_err:
        print(f"⚠️ Failed to cancel order {order_id}: {cancel_err}")
    return latest


# === Google Sheets Helper ===
def load_bot_state() -> None:
    global open_positions, open_positions_data, last_buy_time, last_trade_time, symbol_cooldowns
    if not STATE_PATH or not os.path.exists(STATE_PATH):
        return
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as state_file:
            data = json.load(state_file)
        open_positions = set(data.get("open_positions", []))
        open_positions_data = data.get("open_positions_data", {}) or {}
        last_buy = {k: float(v) for k, v in (data.get("last_buy_time") or {}).items()}
        last_trade = {k: float(v) for k, v in (data.get("last_trade_time") or {}).items()}
        cooldowns = {k: float(v) for k, v in (data.get("symbol_cooldowns") or {}).items()}
        last_buy_time = defaultdict(lambda: 0, last_buy)
        last_trade_time = defaultdict(lambda: 0, last_trade)
        symbol_cooldowns = defaultdict(lambda: 0.0, cooldowns)
        print(f"🗂️ Restored bot state from {STATE_PATH}.")
    except Exception as err:
        print(f"⚠️ Failed to load bot state: {err}")


def save_bot_state() -> None:
    if not STATE_PATH:
        return
    try:
        payload = {
            "open_positions": sorted(open_positions),
            "open_positions_data": open_positions_data,
            "last_buy_time": {k: float(v) for k, v in last_buy_time.items()},
            "last_trade_time": {k: float(v) for k, v in last_trade_time.items()},
            "symbol_cooldowns": {k: float(v) for k, v in symbol_cooldowns.items()},
        }
        with open(STATE_PATH, "w", encoding="utf-8") as state_file:
            json.dump(payload, state_file, ensure_ascii=True, indent=2)
    except Exception as err:
        print(f"⚠️ Failed to persist bot state: {err}")


def get_markets(force: bool = False) -> Dict[str, dict]:
    global markets_cache, markets_last_load
    now = time.time()
    cache_age = now - markets_last_load if markets_last_load else None
    stale = cache_age is not None and cache_age > MARKET_REFRESH_SECONDS
    if force or not markets_cache or stale:
        try:
            markets_cache = call_with_retries(exchange.load_markets)
            markets_last_load = now
        except Exception as err:
            if markets_cache:
                age = now - markets_last_load if markets_last_load else 0
                log_event("warn", "Markets", f"Using stale markets cache (age {age:.0f}s) after refresh failure: {err}")
            else:
                log_event("error", "Markets", f"Failed to load markets: {err}")
                raise
    return markets_cache


def fetch_ohlcv_cached(symbol: str, timeframe: str, limit: int, ttl: Optional[int] = None) -> List[List[float]]:
    if ttl is None:
        ttl = MULTI_TF_CACHE_TTL_SECONDS if timeframe in {"4h", "1d", "1w"} else OHLCV_CACHE_TTL_SECONDS
    key = (symbol, timeframe, limit)
    now = time.time()
    cached = ohlcv_cache.get(key)
    cached_age = (now - cached['ts']) if cached else None
    if cached and cached_age is not None and cached_age < ttl:
        return cached['data']
    try:
        data = call_with_retries(exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
        ohlcv_cache[key] = {'ts': now, 'data': data}
        return data
    except Exception as err:
        if cached:
            log_event("warn", "DataCache", f"Returning stale OHLCV for {symbol} {timeframe} (age {cached_age:.0f}s): {err}")
            return cached['data']
        raise


def fetch_orderbook_metrics(symbol: str, depth_percent: float = DEPTH_BAND_PERCENT) -> Dict[str, float]:
    key = (symbol, depth_percent)
    now = time.time()
    cached = orderbook_cache.get(key)
    cached_age = (now - cached.get('timestamp', 0.0)) if cached else None
    if cached and cached_age is not None and cached_age < ORDERBOOK_CACHE_TTL_SECONDS:
        return cached

    try:
        orderbook = call_with_retries(exchange.fetch_order_book, symbol, limit=DEPTH_ORDERBOOK_LIMIT)
    except Exception as err:
        if cached:
            log_event("warn", "OrderBook", f"Returning stale depth for {symbol} (age {cached_age:.0f}s): {err}")
            return cached
        print(f"⚠️ Failed to fetch order book for {symbol}: {err}")
        return {}

    raw_bids = orderbook.get('bids') or []
    raw_asks = orderbook.get('asks') or []

    def _sanitize_levels(levels: List) -> List[Tuple[float, float]]:
        cleaned: List[Tuple[float, float]] = []
        for level in levels:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                continue
            price = level[0]
            amount = level[1]
            if not isinstance(price, (int, float)) or not isinstance(amount, (int, float)):
                continue
            cleaned.append((float(price), float(amount)))
        return cleaned

    bids = _sanitize_levels(raw_bids)
    asks = _sanitize_levels(raw_asks)

    if not bids or not asks:
        return {}

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    if not isinstance(best_bid, (int, float)) or not isinstance(best_ask, (int, float)) or best_bid <= 0 or best_ask <= 0:
        return {}

    mid_price = (best_bid + best_ask) / 2
    spread_pct = ((best_ask - best_bid) / mid_price) * 100 if mid_price > 0 else None
    band_ratio = max(depth_percent, 0.01) / 100
    lower_bound = mid_price * (1 - band_ratio)
    upper_bound = mid_price * (1 + band_ratio)

    bid_volume = sum(amount for price, amount in bids if price >= lower_bound)
    ask_volume = sum(amount for price, amount in asks if price <= upper_bound)
    depth_ratio = (bid_volume / ask_volume) if ask_volume and ask_volume > 0 else None

    result = {
        'timestamp': now,
        'mid_price': mid_price,
        'spread_pct': spread_pct,
        'bid_volume_band': bid_volume,
        'ask_volume_band': ask_volume,
        'depth_ratio': depth_ratio,
    }
    orderbook_cache[key] = result
    return result


def get_listing_age_days(symbol: str) -> Optional[float]:
    now = time.time()
    cached = listing_age_cache.get(symbol)
    if cached and (now - cached.get('timestamp', 0.0)) < LISTING_AGE_CACHE_SECONDS:
        return cached.get('age_days')

    try:
        daily = fetch_ohlcv_cached(
            symbol,
            timeframe='1d',
            limit=LISTING_AGE_LOOKBACK_DAYS,
            ttl=LISTING_AGE_CACHE_SECONDS,
        )
    except Exception as err:
        print(f"⚠️ Failed to estimate listing age for {symbol}: {err}")
        return None

    if not daily:
        return None

    first_timestamp = daily[0][0]
    if not isinstance(first_timestamp, (int, float)):
        return None

    age_days = max((now - (first_timestamp / 1000)) / 86400, 0.0)
    listing_age_cache[symbol] = {
        'timestamp': now,
        'age_days': age_days,
    }
    return age_days


def reconcile_quote_alias(balance_key: str) -> str:
    if balance_key.startswith('Z') and len(balance_key) > 1:
        return balance_key[1:]
    return balance_key


def pick_quote_balance(free_balances: Dict[str, float]) -> Tuple[str, float]:
    normalized = {reconcile_quote_alias(k.upper()): float(v or 0) for k, v in (free_balances or {}).items()}
    candidates = PREFERRED_QUOTES or [DEFAULT_QUOTE_ASSET]
    if DEFAULT_QUOTE_ASSET not in candidates:
        candidates = [DEFAULT_QUOTE_ASSET] + candidates
    for code in candidates:
        amount = normalized.get(code)
        if amount and amount > 0:
            return code, amount
    return DEFAULT_QUOTE_ASSET, float(normalized.get(DEFAULT_QUOTE_ASSET, 0.0))


def normalize_order_values(symbol: str, price: float, amount: float) -> Tuple[float, float]:
    get_markets()
    precise_price = price
    precise_amount = amount
    try:
        precise_amount = float(exchange.amount_to_precision(symbol, amount))
    except Exception as err:
        print(f"⚠️ amount_to_precision failed for {symbol}: {err}")
    try:
        precise_price = float(exchange.price_to_precision(symbol, price))
    except Exception as err:
        print(f"⚠️ price_to_precision failed for {symbol}: {err}")
    return precise_price, precise_amount


def fetch_market_minimums(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """Return Kraken's minimum notional (cost) and amount for the given pair."""
    get_markets()
    try:
        market = exchange.market(symbol)
    except Exception as err:
        print(f"⚠️ Failed to load market limits for {symbol}: {err}")
        return None, None

    limits = (market or {}).get('limits') or {}
    amount_limits = limits.get('amount') or {}
    cost_limits = limits.get('cost') or {}

    def _cast(value: object) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return _cast(cost_limits.get('min')), _cast(amount_limits.get('min'))


def current_total_exposure() -> float:
    exposure = 0.0
    for info in open_positions_data.values():
        price = info.get('entry_price')
        amount = info.get('amount')
        if isinstance(price, (int, float)) and isinstance(amount, (int, float)):
            exposure += max(price, 0) * max(amount, 0)
    return float(exposure)


if not SKIP_BOT_INIT:
    load_bot_state()
    print("🔄 Initializing markets...", flush=True)
    get_markets(force=True)


@lru_cache(maxsize=1)
def get_gspread_client():
    """Authorize once and reuse the Sheets client to avoid repeated file I/O."""
    if not GOOGLE_CREDENTIALS_PATH:
        return None
    if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
        log_event("warn", "sheets", f"Google credentials not found at {GOOGLE_CREDENTIALS_PATH}; disabling sheet logging.")
        return None
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIALS_PATH, scope)
        return gspread.authorize(creds)
    except Exception as e:
        send_telegram_alert(f"❌ Google Sheets auth error: {str(e)}")
        return None

def log_trade(symbol, side, amount, price, reason):
    timestamp = datetime.now().isoformat()
    # Enforce column order: ["Timestamp", "Symbol", "Side", "Amount", "Price", "Reason"]
    row = [timestamp, symbol, side, amount, price, reason]

    # Log to local CSV
    with open(TRADE_LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Timestamp", "Symbol", "Side", "Amount", "Price", "Reason"])
        writer.writerow(row)

    # Log to backup CSV
    with open(TRADE_LOG_PATH_BACKUP, mode='a', newline='') as backup_file:
        backup_writer = csv.writer(backup_file)
        if backup_file.tell() == 0:
            backup_writer.writerow(["Timestamp", "Symbol", "Side", "Amount", "Price", "Reason"])
        backup_writer.writerow(row)

    # Log to Google Sheet
    try:
        trade_tab = os.getenv("TRADE_SHEET_NAME", "Crypto Trade Log")
        client = get_gspread_client()
        if client:
            sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID"))
            worksheet = sheet.worksheet(trade_tab)
            worksheet.append_row(row)
    except Exception as e:
        send_telegram_alert(f"⚠️ Failed to log trade to Google Sheet tab '{trade_tab}': {str(e)}")

    send_telegram_alert(
        f"📒 LOGGED TRADE: {side.upper()} {symbol} | Amount: {amount} @ {price:.4f} ({reason})"
    )


def log_trade_features(
    symbol: str,
    score: Optional[float],
    indicators: Optional[Dict[str, object]],
    reasons: Optional[List[str]],
    filled_amount: Optional[float],
    executed_price: Optional[float],
    *,
    actual_notional: Optional[float] = None,
    effective_stop_pct: Optional[float] = None,
) -> None:
    indicators = indicators or {}
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        file_exists = os.path.exists(TRADE_FEATURE_LOG_PATH)
        with open(TRADE_FEATURE_LOG_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    "Timestamp",
                    "Symbol",
                    "Score",
                    "FilledAmount",
                    "ExecutedPrice",
                    "ActualNotional",
                    "EffectiveStopPct",
                    "QuoteVolume",
                    "SpreadPct",
                    "RSI1h",
                    "ATRPct",
                    "VolumeRatio",
                    "ADX1h",
                    "MACDHistSlope",
                    "ROCPct",
                    "DepthRatio",
                    "DepthBidNotional",
                    "DepthAskNotional",
                    "TimeframeAlignment",
                    "FreshSignalAge",
                    "ListingAgeDays",
                    "Reasons",
                ])
            row = [
                timestamp,
                symbol,
                score,
                filled_amount,
                executed_price,
                actual_notional,
                effective_stop_pct,
                indicators.get('quote_volume') or indicators.get('quoteVolume') or indicators.get('volUsd24h'),
                indicators.get('spread_pct'),
                indicators.get('rsi_1h') or indicators.get('rsi'),
                indicators.get('atr_pct'),
                indicators.get('volume_ratio'),
                indicators.get('adx_1h'),
                indicators.get('macd_hist_slope'),
                indicators.get('roc_pct'),
                indicators.get('depth_ratio'),
                indicators.get('depth_bid_notional'),
                indicators.get('depth_ask_notional'),
                indicators.get('timeframe_alignment'),
                indicators.get('fresh_signal_age'),
                indicators.get('listing_age_days'),
                " | ".join(reasons or [])
            ]
            writer.writerow(row)
    except Exception as err:
        print(f"⚠️ Failed to log trade features: {err}")

def log_portfolio_snapshot():
    try:
        usd = 0
        total = 0
        try:
            balance = call_with_retries(exchange.fetch_balance)
            balance_total = balance.get('total', {})
            print("💰 Kraken Balance Keys:", list(balance_total.keys()))
            usd = balance_total.get('ZUSD', balance_total.get('USD', 0))
            total = sum(
                v for v in balance_total.values()
                if isinstance(v, (int, float))
            )
        except Exception as e:
            print("❌ Error fetching balance:", e)
        timestamp = datetime.now().isoformat()
        portfolio_value = float(total) if isinstance(total, (int, float)) else 0.0

        ensure_directory(os.path.dirname(EQUITY_LOG_PATH))
        equity_log_exists = os.path.exists(EQUITY_LOG_PATH)
        with open(EQUITY_LOG_PATH, mode='a', newline='') as equity_file:
            equity_writer = csv.writer(equity_file)
            if not equity_log_exists:
                equity_writer.writerow(["timestamp", "cash_balance", "portfolio_value"])
            equity_writer.writerow([timestamp, float(usd or 0.0), portfolio_value])

        # Log to local CSV
        with open(PORTFOLIO_LOG_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, usd, total])
            # Also write to backup CSV
            with open("crypto_portfolio_log_backup.csv", mode='a', newline='') as backup_file:
                backup_writer = csv.writer(backup_file)
                backup_writer.writerow([timestamp, usd, total])

        # Log to Google Sheet
        try:
            client = get_gspread_client()
            if client:
                sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID")).worksheet(os.getenv("PORTFOLIO_SHEET_NAME"))
                sheet.append_row([timestamp, usd, total])
        except Exception as sheet_err:
            send_telegram_alert(f"⚠️ Failed to log portfolio to sheet: {str(sheet_err)}")

        # --- Equity analytics & notifications ---
        now_utc = datetime.now(timezone.utc)

        def load_equity_dataframe(equity_log_file: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
            if not equity_log_file or not os.path.exists(equity_log_file):
                return None, "📈 Daily P&L: awaiting first equity snapshot."
            try:
                df = pd.read_csv(equity_log_file)
            except Exception as err:
                return None, f"⚠️ Error loading equity log: {err}"

            if df.empty or "timestamp" not in df.columns or "portfolio_value" not in df.columns:
                return None, "📈 Daily P&L: awaiting sufficient equity data."

            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors='coerce')
            except Exception as parse_err:
                return None, f"⚠️ Equity log timestamp parse error: {parse_err}"

            df = df.dropna(subset=["timestamp", "portfolio_value"])
            if df.empty:
                return None, "📈 Daily P&L: awaiting sufficient equity data."

            df.sort_values("timestamp", inplace=True)
            return df, None

        def summarize_daily_pnl(df: pd.DataFrame, as_of: datetime) -> str:
            today_logs = df[df["timestamp"].dt.date == as_of.date()]
            samples = len(today_logs)
            if samples < 2:
                return f"📈 Daily P&L: {samples} snapshot(s); collecting more data."

            start_value = float(today_logs.iloc[0]["portfolio_value"] or 0.0)
            end_value = float(today_logs.iloc[-1]["portfolio_value"] or 0.0)
            change = end_value - start_value
            pct_change = (change / start_value * 100) if start_value else 0.0
            high_value = float(today_logs["portfolio_value"].max())
            low_value = float(today_logs["portfolio_value"].min())

            return (
                "📈 Daily P&L Update\n"
                f"Start: ${start_value:.2f} → Current: ${end_value:.2f}\n"
                f"Δ ${change:+.2f} ({pct_change:+.2f}%) | High/Low: ${high_value:.2f} / ${low_value:.2f}\n"
                f"Snapshots logged: {samples}"
            )

        def summarize_weekly_pnl(df: pd.DataFrame, as_of: datetime) -> Optional[str]:
            week_start = as_of - timedelta(days=7)
            week_logs = df[df["timestamp"] >= week_start]
            samples = len(week_logs)
            if samples < 2:
                return None

            start_value = float(week_logs.iloc[0]["portfolio_value"] or 0.0)
            end_value = float(week_logs.iloc[-1]["portfolio_value"] or 0.0)
            change = end_value - start_value
            pct_change = (change / start_value * 100) if start_value else 0.0
            high_value = float(week_logs["portfolio_value"].max())
            low_value = float(week_logs["portfolio_value"].min())

            return (
                "🗓️ Weekly P&L Summary\n"
                f"Period: {week_start.date()} → {as_of.date()}\n"
                f"Start ${start_value:.2f} → End ${end_value:.2f}\n"
                f"Δ ${change:+.2f} ({pct_change:+.2f}%) | High/Low: ${high_value:.2f} / ${low_value:.2f}\n"
                f"Snapshots this week: {samples}"
            )

        equity_df, equity_status_message = load_equity_dataframe(EQUITY_LOG_PATH)
        if equity_df is not None:
            daily_message = summarize_daily_pnl(equity_df, now_utc)
            weekly_message = summarize_weekly_pnl(equity_df, now_utc)
        else:
            daily_message = equity_status_message
            weekly_message = None

        # --- Unrealized P&L summary ---
        unrealized = 0.0
        try:
            for sym, pos in open_positions_data.items():
                ticker = call_with_retries(exchange.fetch_ticker, sym)
                last_price = ticker.get("last") or ticker.get("close") or 0
                if isinstance(last_price, (int, float)):
                    entry_price = pos.get("entry_price", 0)
                    amount = pos.get("amount", 0)
                    unrealized += (last_price - entry_price) * amount
        except Exception as uerr:
            print(f"⚠️ Unrealized P&L fetch error: {uerr}")

        send_telegram_alert(f"💵 Unrealized P&L: ${unrealized:+.2f}")
        if daily_message:
            send_telegram_alert(daily_message)

        global last_weekly_summary_date
        if weekly_message and now_utc.weekday() == 6:  # Sunday summary to avoid noise mid-week
            if last_weekly_summary_date != now_utc.date().isoformat():
                send_telegram_alert(weekly_message)
                last_weekly_summary_date = now_utc.date().isoformat()

        print(f"💾 Snapshot: USD=${usd:.2f}, Total=${total:.2f}")
    except Exception as e:
        send_telegram_alert(f"⚠️ Failed to log portfolio snapshot: {str(e)}")

# === RISK MANAGEMENT & EXECUTION ===
def execute_trade(
    symbol: str,
    side: str,
    amount: float,
    limit_price: Optional[float] = None,
    reason: str = "Live trade executed"
) -> Optional[Dict[str, float]]:
    """Unified trade executor with dry-run & safe fallback."""
    now = time.time()
    if side == "buy":
        if symbol in open_positions:
            reason = f"⛔ Trade rejected: {symbol} is already in open_positions."
            print(reason)
            send_telegram_alert(reason)
            return None
        cooldown_until_ts = symbol_cooldowns.get(symbol, 0.0)
        if cooldown_until_ts and now < cooldown_until_ts:
            wait_min = int((cooldown_until_ts - now) / 60)
            reason = f"⏳ Trade rejected: cooldown active for {symbol} ({wait_min} min remaining)."
            print(reason)
            send_telegram_alert(reason)
            return None
        if now - last_buy_time[symbol] < COOLDOWN_SECONDS:
            wait_time = int(COOLDOWN_SECONDS - (now - last_buy_time[symbol])) // 60
            reason = f"⏳ Trade rejected: cooldown active for {symbol} ({wait_time} min remaining)."
            print(reason)
            send_telegram_alert(reason)
            return None
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            reason = f"🚫 Max open positions reached ({MAX_OPEN_POSITIONS}); skipping {symbol} buy."
            print(reason)
            send_telegram_alert(reason)
            return None
    if side == "buy" and now - last_trade_time[symbol] < TRADE_COOLDOWN_SECONDS:
        wait_min = int((TRADE_COOLDOWN_SECONDS - (now - last_trade_time[symbol])) / 60)
        reason = f"⏳ GLOBAL COOLDOWN: {symbol} trade blocked ({wait_min} min left)."
        print(reason)
        send_telegram_alert(reason)
        return None

    try:
        ticker = call_with_retries(exchange.fetch_ticker, symbol)
        reference_price = extract_valid_price(ticker)
        if reference_price is None:
            message = f"⚠️ No valid price data available for {symbol}; skipping {side} order."
            log_event("warn", "TradeEngine", message)
            return None

        print(f"🛒 Placing {side.upper()} order for {symbol} on {exchange_name.upper()}...")
        if amount is None or amount <= 0:
            message = f"⚠️ Invalid trade amount for {symbol}; skipping {side} order."
            log_event("warn", "TradeEngine", message)
            return None

        try:
            order_book = call_with_retries(exchange.fetch_order_book, symbol, limit=10)
        except Exception as book_err:
            message = f"⚠️ Failed to fetch order book for {symbol}: {book_err}"
            log_event("warn", "TradeEngine", message)
            return None

        side_levels = order_book.get('asks' if side == "buy" else 'bids') or []
        side_levels = [level for level in side_levels if isinstance(level, list) and len(level) >= 2]
        if not side_levels:
            message = f"⚠️ Order book empty for {symbol}; skipping {side} order."
            log_event("warn", "TradeEngine", message)
            return None

        book_price = side_levels[0][0]
        if not isinstance(book_price, (int, float)) or book_price <= 0:
            message = f"⚠️ Order book price invalid for {symbol}; skipping {side} order."
            log_event("warn", "TradeEngine", message)
            return None

        available_volume = sum(level[1] for level in side_levels if isinstance(level[1], (int, float)))
        if side == "sell":
            position = open_positions_data.get(symbol, {})
            position_amount = position.get('amount', 0)
            amount = min(amount, position_amount) if position_amount else amount
            if amount <= 0:
                message = f"⚠️ No position size available for {symbol}; skipping sell."
                log_event("warn", "TradeEngine", message)
                return None

        fallback_price = limit_price if isinstance(limit_price, (int, float)) and limit_price > 0 else reference_price
        buffer = max(LIMIT_PRICE_BUFFER, 0.0)
        if side == "buy":
            limit_price = book_price if buffer == 0 else book_price * (1 + buffer)
            if fallback_price and buffer > 0:
                limit_price = min(limit_price, fallback_price * (1 + buffer))
        else:
            limit_price = book_price if buffer == 0 else book_price * (1 - buffer)
            if fallback_price and buffer > 0:
                limit_price = max(limit_price, fallback_price * (1 - buffer))

        limit_price, precise_amount = normalize_order_values(symbol, limit_price, amount)
        amount = precise_amount
        if limit_price <= 0:
            message = f"⚠️ Computed limit price invalid for {symbol}; skipping {side} order."
            log_event("warn", "TradeEngine", message)
            return None
        if amount <= 0:
            message = f"⚠️ Order size rounded to zero for {symbol}; skipping {side} order."
            log_event("warn", "TradeEngine", message)
            return None

        if available_volume < amount:
            message = (
                f"⚠️ Not enough liquidity to {side} {amount} {symbol}; "
                f"available {available_volume:.4f}."
            )
            log_event("warn", "TradeEngine", message)
            return None

        if side == "buy":
            projected_exposure = current_total_exposure() + (fallback_price * amount)
            if projected_exposure > MAX_TOTAL_EXPOSURE_USD:
                message = (
                    f"🚫 Exposure cap hit: projected {projected_exposure:.2f} exceeds {MAX_TOTAL_EXPOSURE_USD:.2f}; "
                    f"skipping {symbol} buy."
                )
                log_event("warn", "TradeEngine", message)
                return None

        if config.DRY_RUN or not config.ALLOW_TRADING:
            log_event(
                "info",
                "TradeEngine",
                f"DRY-RUN: would execute {side.upper()} {symbol} @ {limit_price:.8f} for {amount:.6f}"
            )
            return {"status": "dry_run", "amount": amount, "limit_price": limit_price}

        # --- Begin retry block for buy logic ---
        if side == "buy":
            try:
                order = exchange.create_order(symbol, "limit", "buy", amount, limit_price)
            except Exception as e:
                if not config.ALLOW_MARKET_FALLBACK:
                    log_event(
                        "warn",
                        "TradeEngine",
                        f"Limit order failed for {symbol} and market fallback disabled: {e}"
                    )
                    return None
                log_event("warn", "TradeEngine", f"Limit order failed; retrying market order: {e}")
                try:
                    order = exchange.create_order(symbol, "market", "buy", amount)
                except Exception as e2:
                    log_event("error", "TradeEngine", f"Market order also failed: {e2}")
                    return None
        elif side == "sell":
            order = exchange.create_order(symbol, "limit", "sell", amount, limit_price)
        else:
            log_event("error", "TradeEngine", f"Invalid trade side '{side}' for {symbol}")
            return None
        # --- End retry block ---

        placed_price = order.get('price') if isinstance(order, dict) else None
        if not placed_price and side == "buy":
            placed_price = limit_price if limit_price else reference_price
        elif not placed_price:
            placed_price = limit_price
        try:
            placed_price_float = float(placed_price) if placed_price else float(reference_price)
        except (TypeError, ValueError):
            placed_price_float = float(reference_price)
        print(f"✅ Trade placed: {symbol} at {placed_price_float:.4f} for qty {amount}", flush=True)

        print(
            f"📍LIMIT {side.upper()} for {symbol} @ {limit_price:.8f} "
            f"(amount: {amount:.6f}) placed; waiting for fill..."
        )
        final_order = wait_for_order_fill(symbol, order)
        status = final_order.get('status')
        filled_amount = float(final_order.get('filled') or 0.0)

        if filled_amount <= 0:
            message = f"⏹️ {symbol} {side} order not filled (status: {status})."
            log_event("warn", "TradeEngine", message)
            return None

        executed_price = final_order.get('average') or final_order.get('price') or fallback_price
        if executed_price is None or executed_price <= 0:
            executed_price = fallback_price

        if executed_price is None or executed_price <= 0:
            message = f"⚠️ {symbol} execution price unavailable; skipping trade log."
            log_event("warn", "TradeEngine", message)
            return None

        executed_price = float(executed_price)
        if filled_amount < amount:
            send_telegram_alert(
                f"ℹ️ Partial fill for {symbol} {side}: filled {filled_amount} of {amount}. Remaining order cancelled."
            )

        last_trade_time[symbol] = now
        symbol_cooldowns[symbol] = now + TRADE_COOLDOWN_SECONDS
        if side == "buy":
            last_buy_time[symbol] = now
            open_positions.add(symbol)
            open_positions_data[symbol] = {
                "entry_price": executed_price,
                "amount": filled_amount,
                "peak_price": executed_price,
            }
        else:
            position = open_positions_data.get(symbol, {})
            if position:
                remaining = max(position.get('amount', 0) - filled_amount, 0)
                if remaining > 1e-8:
                    position['amount'] = remaining
                else:
                    open_positions.discard(symbol)
                    open_positions_data.pop(symbol, None)
            else:
                open_positions.discard(symbol)
                open_positions_data.pop(symbol, None)

        log_trade(symbol, side, filled_amount, executed_price, reason)
        print(f"✅ {side.upper()} order filled for {symbol}: {filled_amount} @ ${executed_price:.4f} (status: {status})")
        save_bot_state()
        return {
            "filled_amount": filled_amount,
            "executed_price": executed_price,
        }
    except (NetworkError, ExchangeError) as e:
        log_event("warn", "TradeEngine", f"Exchange error {type(e).__name__}: {e}")
        return None
    except Exception as e:
        log_event("error", "TradeEngine", f"Unexpected error: {e}")
        return None

# === POSITION MONITORING ===
def monitor_positions():
    print("🔍 Monitoring open positions...")
    state_dirty = False
    for symbol in list(open_positions):
        try:
            ohlcv = fetch_ohlcv_cached(symbol, timeframe='1h', limit=50, ttl=60)
            if not ohlcv or len(ohlcv) < 2:
                print(f"⚠️ Insufficient OHLCV data for {symbol}; skipping monitoring cycle.")
                continue
            closes = [candle[4] for candle in ohlcv]
            highs = [candle[2] for candle in ohlcv]
            lows = [candle[3] for candle in ohlcv]

            tr_list = []
            for i in range(1, len(closes)):
                high = highs[i]
                low = lows[i]
                prev_close = closes[i - 1]
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                tr_list.append(tr)
            if len(tr_list) < 14:
                print(f"⚠️ Not enough data to compute ATR for {symbol}; skipping.")
                continue
            atr_window = [tr for tr in tr_list[-14:] if isinstance(tr, (int, float))]
            if not atr_window:
                print(f"⚠️ ATR window invalid for {symbol}; skipping.")
                continue
            atr = float(np.mean(atr_window))
            if not np.isfinite(atr) or atr <= 0:
                print(f"⚠️ ATR computed as non-positive for {symbol}; skipping.")
                continue

            current_price = closes[-1]
            if not isinstance(current_price, (int, float)) or current_price <= 0:
                print(f"⚠️ Current price invalid for {symbol}; skipping.")
                continue

            entry = open_positions_data.get(symbol, {})
            if not entry:
                continue
            entry_price = entry.get('entry_price')
            amount = entry.get('amount', 0)
            peak_price = entry.get('peak_price') or entry_price
            if not isinstance(entry_price, (int, float)) or entry_price <= 0:
                print(f"⚠️ Entry price invalid for {symbol}; removing from tracking.")
                open_positions.discard(symbol)
                open_positions_data.pop(symbol, None)
                save_bot_state()
                continue
            if amount <= 0:
                print(f"⚠️ Position amount invalid for {symbol}; removing from tracking.")
                open_positions.discard(symbol)
                open_positions_data.pop(symbol, None)
                save_bot_state()
                continue

            change_pct = ((current_price - entry_price) / entry_price) * 100

            if current_price > peak_price:
                entry['peak_price'] = current_price
                peak_price = current_price
                state_dirty = True

            percent_stop_price = entry_price * (1 - HARD_STOP_PERCENT / 100)
            tp_level = entry_price + (ATR_TAKE_PROFIT_MULTIPLIER * atr)
            atr_trailing_level = peak_price - (ATR_TRAIL_MULTIPLIER * atr)
            atr_hard_level = entry_price - (ATR_HARD_STOP_MULTIPLIER * atr)
            atr_trailing_level = max(atr_trailing_level, percent_stop_price)

            atr_stop_pct = (ATR_HARD_STOP_MULTIPLIER * atr / entry_price) * 100
            trail_stop_pct = (
                ((peak_price - atr_trailing_level) / peak_price) * 100
                if peak_price else None
            )

            if change_pct <= -HARD_STOP_PERCENT:
                send_telegram_alert(
                    f"🛑 HARD {HARD_STOP_PERCENT:.2f}% STOP triggered for {symbol} ({change_pct:.2f}%)"
                )
                execute_trade(symbol, "sell", amount, current_price, reason=f"Hard {HARD_STOP_PERCENT:.2f}% stop")
            elif current_price <= atr_hard_level:
                send_telegram_alert(
                    f"🛑 ATR HARD STOP triggered for {symbol} ({change_pct:.2f}%, ATR stop {atr_stop_pct:.2f}%)"
                )
                execute_trade(symbol, "sell", amount, current_price, reason="ATR hard stop")
            elif current_price <= atr_trailing_level:
                send_telegram_alert(
                    f"📉 ATR TRAILING STOP triggered for {symbol} ({change_pct:.2f}%, trail {trail_stop_pct:.2f}% from peak)"
                )
                execute_trade(symbol, "sell", amount, current_price, reason="ATR trailing stop")
            elif current_price >= tp_level:
                send_telegram_alert(
                    f"🎯 ATR TAKE PROFIT HIT for {symbol} (+{change_pct:.2f}%)"
                )
                execute_trade(symbol, "sell", amount, current_price, reason="ATR take profit")
            else:
                print(
                    f"⏳ {symbol} holding: {change_pct:.2f}% | peak ${peak_price:.4f} | "
                    f"hard stop @ ${percent_stop_price:.4f} | atr hard @ ${atr_hard_level:.4f} | "
                    f"trail @ ${atr_trailing_level:.4f} | tp @ ${tp_level:.4f}"
                )
        except Exception as e:
            send_telegram_alert(f"⚠️ monitor_positions error for {symbol}: {str(e)}")
    if state_dirty:
        save_bot_state()

# === MARKET SCANNER ===
def log_scanner_snapshot(records: List[Dict]) -> None:
    if not records:
        return
    try:
        timestamp = datetime.now().isoformat()
        file_exists = os.path.exists(SCANNER_SNAPSHOT_PATH)
        with open(SCANNER_SNAPSHOT_PATH, mode='a', newline='') as log_file:
            writer = csv.writer(log_file)
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "symbol",
                    "score",
                    "quote_volume",
                    "spread_pct",
                    "depth_ratio",
                    "depth_bid_notional",
                    "depth_ask_notional",
                    "atr_pct",
                    "rsi_1h",
                    "adx_1h",
                    "volume_ratio",
                    "macd_hist_slope",
                    "roc_pct",
                    "multi_timeframe_alignment",
                    "fresh_signal_age",
                    "listing_age_days",
                    "reasons"
                ])
            for record in records:
                indicators = record.get('indicators', {})
                writer.writerow([
                    timestamp,
                    record.get('symbol'),
                    record.get('score'),
                    indicators.get('quote_volume'),
                    indicators.get('spread_pct'),
                    indicators.get('depth_ratio'),
                    indicators.get('depth_bid_notional'),
                    indicators.get('depth_ask_notional'),
                    indicators.get('atr_pct'),
                    indicators.get('rsi_1h'),
                    indicators.get('adx_1h'),
                    indicators.get('volume_ratio'),
                    indicators.get('macd_hist_slope'),
                    indicators.get('roc_pct'),
                    indicators.get('timeframe_alignment'),
                    indicators.get('fresh_signal_age'),
                    indicators.get('listing_age_days'),
                    " | ".join(record.get('reasons', []))
                ])
    except Exception as err:
        print(f"⚠️ Failed to persist scanner snapshot: {err}")


def log_near_misses(entries: List[Dict[str, object]]) -> None:
    if not entries:
        return
    try:
        file_exists = os.path.exists(SCANNER_NEAR_MISS_PATH)
        fieldnames = [
            "timestamp",
            "symbol",
            "reason",
            "quote_volume",
            "volume_threshold",
            "soft_volume_threshold",
            "spread_pct",
            "max_spread_percent",
            "soft_spread_limit",
            "depth_ratio",
            "min_depth_imbalance",
            "soft_depth_ratio",
            "atr_pct",
            "regime_state"
        ]
        with open(SCANNER_NEAR_MISS_PATH, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            for entry in entries:
                writer.writerow(entry)
    except Exception as err:
        print(f"⚠️ Failed to persist near-miss log: {err}")


def extract_quote_volume(ticker: Dict) -> Optional[float]:
    if not isinstance(ticker, dict):
        return None
    for key in ("quoteVolume", "baseVolume", "volume", "volCcy24h", "volUsd24h", "vol24h"):
        value = ticker.get(key)
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    info = ticker.get('info') if isinstance(ticker.get('info'), dict) else {}
    for key in ("quoteVolume", "volCcy24h", "volUsd24h", "vol24h"):
        value = info.get(key)
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def scan_top_cryptos(
    regime_state: RegimeState,
    limit: int = 5,
    quote_asset: Optional[str] = None,
    min_volume: Optional[float] = None,
    allow_fallback: bool = True,
) -> List[Dict[str, object]]:
    try:
        quote_asset = (quote_asset or DEFAULT_QUOTE_ASSET).upper()
        print(f"🔍 Scanning {exchange_name.upper()} {quote_asset} markets with enhanced filters (Regime Score: {regime_state.score})...")
        markets = get_markets()
        candidates: List[Dict[str, object]] = []
        symbol_suffix = f"/{quote_asset}"

        volume_threshold = max(0.0, min_volume if min_volume is not None else MIN_QUOTE_VOLUME_24H)
        fallback_volume = MIN_QUOTE_VOLUME_FALLBACK
        if volume_threshold <= 0:
            volume_threshold = 0.0
        if fallback_volume <= 0 or fallback_volume >= volume_threshold:
            fallback_volume = volume_threshold
        if min_volume is not None and min_volume < MIN_QUOTE_VOLUME_24H:
            print(
                f"ℹ️ Using relaxed volume threshold {volume_threshold:,.0f} "
                f"(default {MIN_QUOTE_VOLUME_24H:,.0f})."
            )

        soft_volume_threshold = max(fallback_volume, volume_threshold * SOFT_VOLUME_MULTIPLIER)
        soft_spread_limit = MAX_SPREAD_PERCENT + SOFT_SPREAD_BUFFER
        soft_depth_ratio = max(0.0, MIN_DEPTH_IMBALANCE - SOFT_DEPTH_BUFFER)
        near_miss_entries: List[Dict[str, object]] = []

        def record_near_miss(symbol_name: str, reason: str, extra: Optional[Dict[str, object]] = None) -> None:
            payload: Dict[str, object] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol_name,
                "reason": reason,
                "volume_threshold": volume_threshold,
                "soft_volume_threshold": soft_volume_threshold,
                "max_spread_percent": MAX_SPREAD_PERCENT,
                "soft_spread_limit": soft_spread_limit,
                "min_depth_imbalance": MIN_DEPTH_IMBALANCE,
                "soft_depth_ratio": soft_depth_ratio,
                "regime_state": regime_state.status,
            }
            if extra:
                payload.update(extra)
            near_miss_entries.append(payload)

        restricted_keywords = ["RETARDIO", "SPICE", "KERNEL", "HIPPO", "MERL", "DEGEN", "BMT"]

        tickers = list(markets.items())
        print(f"🔍 Starting scan: {len(tickers)} tokens to evaluate...", flush=True)

        def log_skip(symbol_name: str, reason: str) -> None:
            print(f"🚫 Skipping {symbol_name} — {reason}", flush=True)

        for symbol, market in tickers:
            if symbol.upper() in RESTRICTED_SYMBOLS or symbol.upper() in RUNTIME_RESTRICTED_SYMBOLS:
                # Silent skip for blacklist to reduce noise, unless DEBUG
                # log_skip(symbol, "restricted symbol")
                continue
            market_quote = (market or {}).get('quote')
            if market_quote:
                if str(market_quote).upper() != quote_asset:
                    continue
            elif not symbol.endswith(symbol_suffix):
                continue
            if any(keyword in symbol for keyword in restricted_keywords):
                log_skip(symbol, "symbol flagged by restricted keywords")
                continue
            if market and not market.get('active', True):
                log_skip(symbol, "market not active")
                continue

            try:
                ticker = call_with_retries(exchange.fetch_ticker, symbol)
            except Exception as fetch_err:
                err_str = str(fetch_err).lower()
                if "restricted" in err_str or "permission denied" in err_str or "service unavailable" in err_str:
                    print(f"⚠️ Flagging {symbol} as restricted/unavailable: {fetch_err}")
                    RUNTIME_RESTRICTED_SYMBOLS.add(symbol.upper())
                log_skip(symbol, f"ticker fetch failed ({fetch_err})")
                continue

            quote_volume = extract_quote_volume(ticker)
            if quote_volume is None:
                record_near_miss(symbol, "volume unavailable", {"quote_volume": 0.0})
                log_skip(symbol, "failed volume filter (unavailable)")
                continue
            if quote_volume < soft_volume_threshold:
                record_near_miss(
                    symbol,
                    "volume below soft floor",
                    {"quote_volume": quote_volume}
                )
                log_skip(
                    symbol,
                    f"failed volume filter ({quote_volume:.2f} < soft min {soft_volume_threshold:.2f})"
                )
                continue
            volume_penalty = 0.0
            volume_note = ""
            if quote_volume < volume_threshold and volume_threshold > 0:
                deficit_pct = 1 - (quote_volume / volume_threshold)
                volume_penalty = deficit_pct * SCORING_WEIGHTS.get("volume_penalty", 0.0)
                volume_note = f"Volume deficit {deficit_pct*100:.1f}%"
                record_near_miss(
                    symbol,
                    "volume below preferred threshold",
                    {"quote_volume": quote_volume}
                )

            bid = ticker.get('bid')
            ask = ticker.get('ask')
            if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)):
                record_near_miss(symbol, "bid/ask invalid", {"quote_volume": quote_volume})
                log_skip(symbol, "bid/ask not numeric")
                continue
            if bid <= 0 or ask <= 0 or ask <= bid:
                record_near_miss(symbol, "bid/ask invalid spread", {"quote_volume": quote_volume})
                log_skip(symbol, "invalid bid/ask spread")
                continue
            mid_price = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid_price) * 100 if mid_price > 0 else None
            if spread_pct is None:
                record_near_miss(symbol, "spread unavailable", {"quote_volume": quote_volume})
                log_skip(symbol, "spread unavailable")
                continue
            if spread_pct > soft_spread_limit:
                record_near_miss(
                    symbol,
                    "spread above soft limit",
                    {"quote_volume": quote_volume, "spread_pct": spread_pct}
                )
                log_skip(symbol, f"failed spread filter ({spread_pct:.2f}% > soft {soft_spread_limit:.2f}%)")
                continue
            spread_penalty = 0.0
            spread_note = ""
            if spread_pct > MAX_SPREAD_PERCENT:
                over_pct = (spread_pct - MAX_SPREAD_PERCENT) / MAX_SPREAD_PERCENT if MAX_SPREAD_PERCENT > 0 else 0.0
                spread_penalty = over_pct * SCORING_WEIGHTS.get("spread_penalty", 0.0)
                spread_note = f"Spread penalty {over_pct*100:.1f}%"
                record_near_miss(
                    symbol,
                    "spread above preferred threshold",
                    {"quote_volume": quote_volume, "spread_pct": spread_pct}
                )

            depth_metrics = fetch_orderbook_metrics(symbol)
            if not depth_metrics:
                record_near_miss(
                    symbol,
                    "order book metrics unavailable",
                    {"quote_volume": quote_volume, "spread_pct": spread_pct}
                )
                log_skip(symbol, "order book metrics unavailable")
                continue
            depth_spread_pct = depth_metrics.get('spread_pct')
            if isinstance(depth_spread_pct, (int, float)) and depth_spread_pct > MAX_SPREAD_PERCENT:
                log_skip(symbol, f"order book spread high ({depth_spread_pct:.2f}% > {MAX_SPREAD_PERCENT:.2f}%)")
                continue
            depth_ratio = depth_metrics.get('depth_ratio')
            if depth_ratio is None:
                record_near_miss(
                    symbol,
                    "depth ratio unavailable",
                    {"quote_volume": quote_volume, "spread_pct": spread_pct}
                )
                log_skip(symbol, "failed depth imbalance (unavailable)")
                continue
            if depth_ratio < soft_depth_ratio:
                record_near_miss(
                    symbol,
                    "depth ratio below soft floor",
                    {"quote_volume": quote_volume, "spread_pct": spread_pct, "depth_ratio": depth_ratio}
                )
                log_skip(symbol, f"failed depth imbalance ({depth_ratio:.2f} < soft {soft_depth_ratio:.2f})")
                continue
            depth_penalty = 0.0
            depth_note = ""
            if depth_ratio < MIN_DEPTH_IMBALANCE:
                deficit_ratio = (MIN_DEPTH_IMBALANCE - depth_ratio) / MIN_DEPTH_IMBALANCE if MIN_DEPTH_IMBALANCE > 0 else 0.0
                depth_penalty = deficit_ratio * SCORING_WEIGHTS.get("depth_penalty", 0.0)
                depth_note = f"Depth deficit {deficit_ratio*100:.1f}%"
                record_near_miss(
                    symbol,
                    "depth ratio below preferred threshold",
                    {"quote_volume": quote_volume, "spread_pct": spread_pct, "depth_ratio": depth_ratio}
                )
            depth_mid_price = depth_metrics.get('mid_price') or mid_price
            bid_band_volume = float(depth_metrics.get('bid_volume_band') or 0.0)
            ask_band_volume = float(depth_metrics.get('ask_volume_band') or 0.0)
            bid_band_notional = bid_band_volume * depth_mid_price
            ask_band_notional = ask_band_volume * depth_mid_price
            if bid_band_notional < MIN_DEPTH_NOTIONAL_USD or ask_band_notional < MIN_DEPTH_NOTIONAL_USD:
                record_near_miss(
                    symbol,
                    "order book notional too low",
                    {
                        "quote_volume": quote_volume,
                        "spread_pct": spread_pct,
                        "depth_ratio": depth_ratio
                    }
                )
                log_skip(
                    symbol,
                    f"order book notional too low (bid ${bid_band_notional:.2f}, ask ${ask_band_notional:.2f})"
                )
                continue

            try:
                ohlcv_1h = fetch_ohlcv_cached(symbol, timeframe='1h', limit=200)
            except Exception as ohlcv_err:
                log_skip(symbol, f"1h OHLCV fetch failed ({ohlcv_err})")
                continue
            if not ohlcv_1h or len(ohlcv_1h) < 100:
                log_skip(symbol, "insufficient 1h OHLCV history")
                continue

            closes_1h = [c[4] for c in ohlcv_1h]
            highs_1h = [c[2] for c in ohlcv_1h]
            lows_1h = [c[3] for c in ohlcv_1h]
            volumes_1h = [c[5] for c in ohlcv_1h]

            if not closes_1h or closes_1h[-1] is None:
                log_skip(symbol, "invalid close data")
                continue
            last_price = float(closes_1h[-1])
            if last_price <= 0:
                log_skip(symbol, "last price non-positive")
                continue

            ema9_series = calculate_ema_series(closes_1h, 9)
            ema21_series = calculate_ema_series(closes_1h, 21)
            ema50_series = calculate_ema_series(closes_1h, 50)
            if not ema9_series or not ema21_series or not ema50_series:
                log_skip(symbol, "EMA series unavailable")
                continue
            ema9 = ema9_series[-1]
            ema21 = ema21_series[-1]
            ema50 = ema50_series[-1]

            rsi_values = calculate_rsi(closes_1h)
            if not rsi_values:
                log_skip(symbol, "RSI calculation failed")
                continue
            rsi_1h = rsi_values[-1]

            adx_values = calculate_adx(highs_1h, lows_1h, closes_1h)
            adx_1h = adx_values[-1] if adx_values else None

            atr_value = compute_atr(highs_1h, lows_1h, closes_1h)
            atr_pct = (atr_value / last_price) if atr_value else None
            if atr_pct and atr_pct > MAX_ATR_PERCENT:
                record_near_miss(
                    symbol,
                    "atr volatility high",
                    {
                        "quote_volume": quote_volume,
                        "spread_pct": spread_pct,
                        "depth_ratio": depth_ratio,
                        "atr_pct": atr_pct * 100
                    }
                )
                log_skip(symbol, f"failed volatility filter (ATR {atr_pct*100:.2f}% > {MAX_ATR_PERCENT*100:.2f}% max)")
                continue

            avg_vol_20 = np.mean(volumes_1h[-20:]) if len(volumes_1h) >= 20 else None
            volume_ratio = (volumes_1h[-1] / avg_vol_20) if avg_vol_20 else None

            macd_line, signal_line, histogram = compute_macd(closes_1h)
            macd_hist_slope = None
            if histogram and len(histogram) >= 2:
                macd_hist_slope = histogram[-1] - histogram[-2]

            roc_pct = rate_of_change(closes_1h, period=6)

            diff_series = None
            fresh_signal_age = None
            if ema9_series and ema21_series:
                diff_series = np.array(ema9_series) - np.array(ema21_series)
                max_lookback = min(len(diff_series) - 1, FRESH_SIGNAL_LOOKBACK + 5)
                for lookback in range(1, max_lookback + 1):
                    idx = -lookback
                    prev_idx = idx - 1
                    if abs(prev_idx) > len(diff_series):
                        break
                    if diff_series[idx] > 0 and diff_series[prev_idx] <= 0:
                        fresh_signal_age = lookback - 1
                        break

            fresh_signal = fresh_signal_age is not None and fresh_signal_age <= FRESH_SIGNAL_LOOKBACK

            timeframe_alignment_flags = []
            try:
                ohlcv_4h = fetch_ohlcv_cached(symbol, timeframe='4h', limit=120)
            except Exception:
                ohlcv_4h = []
            try:
                ohlcv_1d = fetch_ohlcv_cached(symbol, timeframe='1d', limit=120)
            except Exception:
                ohlcv_1d = []

            def _trend_ok(ohlcv: List[List[float]]) -> bool:
                if not ohlcv or len(ohlcv) < 60:
                    return False
                closes_tf = [c[4] for c in ohlcv]
                ema9_tf = calculate_ema_series(closes_tf, 9)
                ema21_tf = calculate_ema_series(closes_tf, 21)
                if not ema9_tf or not ema21_tf:
                    return False
                return ema9_tf[-1] > ema21_tf[-1]

            if _trend_ok(ohlcv_4h):
                timeframe_alignment_flags.append('4h')
            if _trend_ok(ohlcv_1d):
                timeframe_alignment_flags.append('1d')

            listing_age_days = get_listing_age_days(symbol)
            is_new_listing = (
                listing_age_days is not None and listing_age_days <= NEW_LISTING_BOOST_DAYS
            )

            score = 0.0
            reasons: List[str] = []
            penalty_notes: List[str] = []
            penalty_total = volume_penalty + spread_penalty + depth_penalty

            for note in (volume_note, spread_note, depth_note):
                if note:
                    penalty_notes.append(note)

            if regime_state.status == "BULLISH":
                score += SCORING_WEIGHTS.get("regime_bull", 0.0)
                reasons.append("Regime aligned (bullish bias)")
            elif regime_state.status == "BEARISH_DEFENSIVE":
                penalty_total += SCORING_WEIGHTS.get("regime_bear", 0.0)
                penalty_notes.append(f"Regime headwind: {regime_state.reason}")

            if ema9 > ema21 > ema50:
                stack_weight = SCORING_WEIGHTS.get("ema_stack", 0.0)
                score += stack_weight
                reasons.append(f"1h EMA stack bullish (+{stack_weight:.2f})")

            if fresh_signal and fresh_signal_age is not None:
                freshness = max(0.0, (FRESH_SIGNAL_LOOKBACK + 1 - fresh_signal_age) / (FRESH_SIGNAL_LOOKBACK + 1))
                boost = freshness * SCORING_WEIGHTS.get("fresh_cross", 0.0)
                score += boost
                reasons.append(f"Fresh EMA cross ({fresh_signal_age} bars ago, +{boost:.2f})")

            if isinstance(rsi_1h, (int, float)):
                if 35 <= rsi_1h <= 80:
                    center = 60.0
                    alignment = max(0.0, 1 - abs(rsi_1h - center) / 25.0)
                    if alignment > 0:
                        rsi_boost = alignment * SCORING_WEIGHTS.get("rsi", 0.0)
                        score += rsi_boost
                        reasons.append(f"RSI alignment ({rsi_1h:.1f}, +{rsi_boost:.2f})")
                elif rsi_1h < 30:
                    # Oversold condition - align with Regime improvement (Catch the bounce)
                    bounce_boost = SCORING_WEIGHTS.get("rsi", 1.2) * 1.5
                    score += bounce_boost
                    reasons.append(f"RSI Oversold/Bounce ({rsi_1h:.1f}, +{bounce_boost:.2f})")
                else:
                    # Overbought (>80) or weak chop (30-35)
                    rsi_penalty = (min(abs(rsi_1h - 60.0), 40.0) / 40.0) * SCORING_WEIGHTS.get("rsi_penalty", 0.0)
                    penalty_total += rsi_penalty
                    penalty_notes.append(f"RSI penalty ({rsi_1h:.1f}, -{rsi_penalty:.2f})")

            if volume_ratio and volume_ratio >= 1.0:
                volume_strength = min(volume_ratio - 1.0, 2.5)
                if volume_strength > 0:
                    volume_boost = volume_strength * SCORING_WEIGHTS.get("volume_ratio", 0.0)
                    score += volume_boost
                    reasons.append(f"Volume impulse x{volume_ratio:.2f} (+{volume_boost:.2f})")

            if adx_1h and adx_1h >= 20:
                adx_strength = min((adx_1h - 20) / 25.0, 1.0)
                if adx_strength > 0:
                    adx_boost = adx_strength * SCORING_WEIGHTS.get("adx", 0.0)
                    score += adx_boost
                    reasons.append(f"ADX trend {adx_1h:.1f} (+{adx_boost:.2f})")

            if macd_hist_slope is not None:
                if macd_hist_slope > 0:
                    macd_strength = min(macd_hist_slope, 0.03) / 0.03
                    macd_boost = macd_strength * SCORING_WEIGHTS.get("macd", 0.0)
                    score += macd_boost
                    reasons.append(f"MACD momentum rising (+{macd_boost:.2f})")
                else:
                    macd_penalty = min(abs(macd_hist_slope), 0.03) / 0.03 * (SCORING_WEIGHTS.get("macd", 0.0) / 2)
                    penalty_total += macd_penalty
                    penalty_notes.append(f"MACD momentum fading (-{macd_penalty:.2f})")

            if roc_pct is not None:
                if roc_pct > 0:
                    roc_strength = min(roc_pct / 2.0, 1.0)
                    roc_boost = roc_strength * SCORING_WEIGHTS.get("roc", 0.0)
                    score += roc_boost
                    reasons.append(f"ROC positive ({roc_pct:.2f}%, +{roc_boost:.2f})")
                else:
                    roc_penalty = min(abs(roc_pct) / 2.0, 1.0) * (SCORING_WEIGHTS.get("roc", 0.0) / 2)
                    penalty_total += roc_penalty
                    penalty_notes.append(f"ROC negative ({roc_pct:.2f}%, -{roc_penalty:.2f})")

            if timeframe_alignment_flags:
                tf_boost = len(timeframe_alignment_flags) * SCORING_WEIGHTS.get("timeframe_alignment", 0.0)
                score += tf_boost
                reasons.append(f"Multi-timeframe uptrend ({', '.join(timeframe_alignment_flags)}, +{tf_boost:.2f})")

            if depth_ratio:
                depth_strength = max(0.0, min(depth_ratio / max(MIN_DEPTH_IMBALANCE, 1e-6), 1.5) - 1.0)
                if depth_strength > 0:
                    depth_boost = depth_strength * SCORING_WEIGHTS.get("depth", 0.0)
                    score += depth_boost
                    reasons.append(
                        f"Order-book support {depth_ratio:.2f}x (+{depth_boost:.2f})"
                    )

            if is_new_listing:
                listing_boost = SCORING_WEIGHTS.get("new_listing", 0.0)
                score += listing_boost
                reasons.append(f"Fresh Kraken listing (~{listing_age_days:.1f}d, +{listing_boost:.2f})")

            # Blue Chip Boost
            base_currency = symbol.split('/')[0]
            if base_currency in BLUE_CHIP_SYMBOLS:
                # 20% boost for blue chips
                bc_boost = 1.5 
                score += bc_boost
                reasons.append(f"Blue Chip Asset (+{bc_boost:.2f})")

            ema_slope = (ema9 - ema21) / ema21 if ema21 else 0.0
            if ema_slope > 0:
                slope_strength = min(ema_slope / 0.03, 1.0)
                slope_boost = slope_strength * SCORING_WEIGHTS.get("ema_slope", 0.0)
                score += slope_boost
                reasons.append(f"EMA slope {ema_slope:.4f} (+{slope_boost:.2f})")

            if atr_pct is not None and MAX_ATR_PERCENT > 0:
                stability = max(0.0, 1 - (atr_pct / MAX_ATR_PERCENT))
                if stability > 0:
                    volatility_boost = stability * SCORING_WEIGHTS.get("volatility", 0.0)
                    score += volatility_boost
                    reasons.append(f"Contained volatility ({atr_pct*100:.2f}%, +{volatility_boost:.2f})")

            # === SNIPER LOGIC ===
            # Detect high-quality counter-trend setups (Oversold + Volume)
            is_sniper_setup = False
            if isinstance(rsi_1h, (int, float)) and rsi_1h < 30:
                 # Check for volume capitulation/strength
                 if volume_ratio and volume_ratio > 1.2:
                     is_sniper_setup = True
            
            if is_sniper_setup:
                # WAIVE PENALTIES for trend/regime because we are betting on a reversal
                reasons.append("🎯 SNIPER SETUP DETECTED (Penalties Waived)")
                # Applying a massive boost to ensure it clears the 4.0 hurdle if everything else is decent
                sniper_boost = 3.0
                score += sniper_boost
                reasons.append(f"Sniper Boost (+{sniper_boost:.1f})")
                
                # We do NOT add penalty_total to the final score calculation for these setups
                # effectively ignoring trend headwinds
                final_score = max(score, 0.0) 
            else:
                # Standard Logic
                final_score = max(score - penalty_total, 0.0)
                if penalty_notes:
                    reasons.extend(f"⚠️ {note}" for note in penalty_notes)

            volatility = atr_pct or 0.0
            print(
                f"🔬 {symbol} — Score: {final_score:.2f} | EMA Slope: {ema_slope:.4f} | "
                f"RSI: {rsi_1h:.2f} | Volatility: {volatility:.4f}",
                flush=True
            )

            if final_score < MIN_CANDIDATE_SCORE:
                print(
                    f"⏭️ Skipped {symbol}: Did not pass smart-score filter "
                    f"(score={final_score:.2f} < {MIN_CANDIDATE_SCORE:.2f})",
                    flush=True
                )
                continue

            candidate = {
                "symbol": symbol,
                "score": final_score,
                "model_score": final_score,
                "raw_score": score,
                "ema_slope": ema_slope,
                "rsi": rsi_1h,
                "adx": adx_1h,
                "atr": atr_value,
                "atr_pct": atr_pct,
                "volatility": volatility,
                "depth_ratio": depth_ratio,
                "bid_ask_spread": spread_pct,
                "price": last_price,
                "quote_volume": quote_volume,
                "reasons": reasons,
                "indicators": {
                    "last_price": last_price,
                    "rsi_1h": rsi_1h,
                    "adx_1h": adx_1h,
                    "volume_ratio": volume_ratio,
                    "macd_hist_slope": macd_hist_slope,
                    "roc_pct": roc_pct,
                    "atr_pct": atr_pct,
                    "quote_volume": quote_volume,
                    "spread_pct": spread_pct,
                    "depth_ratio": depth_ratio,
                    "depth_bid_notional": bid_band_notional,
                    "depth_ask_notional": ask_band_notional,
                    "depth_band_percent": DEPTH_BAND_PERCENT,
                    "fresh_signal_age": fresh_signal_age,
                    "timeframe_alignment": ",".join(timeframe_alignment_flags) if timeframe_alignment_flags else "",
                    "listing_age_days": listing_age_days,
                    "penalties": "; ".join(penalty_notes) if penalty_notes else "",
                    "volume_penalty": volume_penalty,
                    "spread_penalty": spread_penalty,
                    "depth_penalty": depth_penalty,
                    "regime_ok": regime_state.can_trade,
                    "regime_reason": regime_state.reason,
                    "model_score": final_score,
                    "raw_score": score,
                }
            }
            candidates.append(candidate)

            # --- Append scanner evaluation log ---
            # Use variables: symbol, last_price, rsi_1h, ema9, ema21, quote_volume, adx_1h, macd_hist_slope, roc_pct
            ema_fast = ema9
            ema_slow = ema21
            price = last_price
            rsi = rsi_1h
            volume = quote_volume
            adx = adx_1h if adx_1h is not None else 0.0
            macd = macd_hist_slope if macd_hist_slope is not None else 0.0
            roc = roc_pct if roc_pct is not None else 0.0
            try:
                log_exists = os.path.exists(SCANNER_LOG_PATH)
                with open(SCANNER_LOG_PATH, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if not log_exists:
                        writer.writerow([
                            "timestamp",
                            "symbol",
                            "price",
                            "rsi",
                            "ema_fast",
                            "ema_slow",
                            "quote_volume",
                            "adx",
                            "macd_slope",
                            "roc_pct"
                        ])
                    writer.writerow([
                        datetime.now(timezone.utc).isoformat(),
                        symbol,
                        f"{price:.4f}",
                        f"{rsi:.2f}",
                        f"{ema_fast:.2f}",
                        f"{ema_slow:.2f}",
                        f"{volume:.2f}",
                        f"{adx:.2f}",
                        f"{macd:.2f}",
                        f"{roc:.2f}"
                    ])
            except Exception as log_err:
                print(f"⚠️ Error writing scanner evaluation log: {log_err}")

        # Debug: Show top 5 scoring cryptos
        if candidates:
            top_scores = sorted(
                (
                    (candidate.get('symbol'), candidate.get('score'))
                    for candidate in candidates
                    if isinstance(candidate, dict) and candidate.get('score') is not None
                ),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for symbol, score in top_scores:
                print(f"🔍 Top candidate: {symbol} | Score: {score}")

        if not candidates:
            log_near_misses(near_miss_entries)
            if allow_fallback and fallback_volume < volume_threshold:
                print(
                    f"ℹ️ No candidates found with volume ≥ {volume_threshold:,.0f}. "
                    f"Retrying scan with relaxed threshold {fallback_volume:,.0f}."
                )
                return scan_top_cryptos(
                    limit=limit,
                    quote_asset=quote_asset,
                    min_volume=fallback_volume,
                    allow_fallback=False,
                    regime_state=regime_state
                )
            return []

        candidates.sort(
            key=lambda item: (
                -(item.get('score') or 0),
                -(item.get('quote_volume') or item.get('indicators', {}).get('quote_volume') or 0)
            )
        )
        top_candidates = candidates[:limit]
        log_scanner_snapshot(top_candidates)
        log_near_misses(near_miss_entries)
        formatted_picks = [(c['symbol'], f"{c['score']:.2f}") for c in top_candidates]
        print("✅ Scanner picks:", formatted_picks)
        return top_candidates
    except Exception as e:
        send_telegram_alert(f"❌ Enhanced scan error: {e}")
        return []


def validate_entry_conditions(candidate: Optional[Dict[str, object]]) -> bool:
    if not candidate:
        return False
    score_ok = candidate.get("score", 0) >= MIN_CANDIDATE_SCORE
    rsi = candidate.get("rsi")
    rsi_ok = isinstance(rsi, (int, float)) and (rsi < 30 or 35 <= rsi <= 75)
    return score_ok and rsi_ok


def evaluate_market_regime(force: bool = False) -> RegimeState:
    global regime_last_check, regime_last_result, last_regime_metrics
    now = time.time()
    if not force and (now - regime_last_check) < REGIME_CACHE_SECONDS and regime_last_result.status != "INIT":
        return regime_last_result

    # 1. Fetch Regime Data (BTC/USD usually)
    minimum_required = max(REGIME_LOOKBACK // 2, REGIME_EMA_SLOW * 4)
    try:
        ohlcv = fetch_ohlcv_cached(
            REGIME_SYMBOL,
            timeframe=REGIME_TIMEFRAME,
            limit=max(REGIME_LOOKBACK, minimum_required),
            ttl=REGIME_CACHE_SECONDS,
        )
    except Exception as err:
        regime_last_check = now
        # Return previous state if valid, else defensive
        if regime_last_result.status != "INIT":
            return regime_last_result
        return RegimeState(0.0, "ERROR", f"Fetch failed: {err}", False, 0.0)

    if not ohlcv or len(ohlcv) < minimum_required:
        regime_last_check = now
        return RegimeState(0.0, "ERROR", "Insufficient regime OHLCV", False, 0.0)

    closes = [c[4] for c in ohlcv if isinstance(c[4], (int, float)) and c[4] > 0]
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    
    # 2. Calculate Indicators
    ema_fast_series = calculate_ema_series(closes, REGIME_EMA_FAST)
    ema_slow_series = calculate_ema_series(closes, REGIME_EMA_SLOW)
    ema_200_series = calculate_ema_series(closes, 200)
    
    if not ema_fast_series or not ema_slow_series:
        regime_last_check = now
        return RegimeState(0.0, "ERROR", "Insufficient EMA data", False, 0.0)

    current_price = closes[-1]
    ema_fast = ema_fast_series[-1]
    ema_slow = ema_slow_series[-1]
    ema_200 = ema_200_series[-1] if ema_200_series else current_price
    
    # ROC
    roc_period = 10
    roc_value = rate_of_change(closes, period=roc_period) or 0.0
    
    # RSI
    rsi_vals = calculate_rsi(closes)
    rsi = rsi_vals[-1] if rsi_vals else 50.0
    
    # ADX
    adx_vals = calculate_adx(highs, lows, closes)
    adx = adx_vals[-1] if adx_vals else 20.0

    # 3. Calculate Score (0-100)
    score = 0.0
    reasons = []

    # A. EMA Structure (Max 30 pts)
    if ema_fast > ema_slow:
        score += 30
        reasons.append("EMA Fast>Slow (+30)")
    else:
        # Penalize dead cross
        reasons.append("EMA Fast<Slow")

    # B. Long Term Trend (Max 20 pts)
    if current_price > ema_200:
        score += 20
        reasons.append("Price>EMA200 (+20)")
    else:
        reasons.append("Price<EMA200")

    # D. RSI Health (Max 10 pts)
    is_oversold = rsi < 30
    if 45 <= rsi <= 75:
        score += 10
        reasons.append(f"RSI Strong {rsi:.1f} (+10)")
    elif is_oversold:
        score += 20 
        reasons.append(f"RSI Oversold {rsi:.1f} (+20)") # Potential bounce
    elif rsi > 80:
        reasons.append(f"RSI Overbought {rsi:.1f}")
        
    # C. Momentum (ROC) (Max 20 pts)
    if roc_value > 0:
        roc_pts = min(roc_value * 2, 20)  # Cap at 20 (e.g., 10% ROC = 20pts)
        score += roc_pts
        reasons.append(f"ROC +{roc_value:.2f}% (+{roc_pts:.1f})")
    elif roc_value > -2.0:
        # Mild negative ROC
        pass
    else:
        # Strong negative ROC penalty (waived if oversold)
        if not is_oversold:
            score -= 10
            reasons.append(f"ROC -{roc_value:.2f}% (-10)")
        else:
            reasons.append(f"ROC -{roc_value:.2f}% (Penalty waived due to Oversold)")

    # E. Trend Strength (ADX) (Max 20 pts)
    if adx > 25:
        # Only bullish if trend is up (EMA aligned), otherwise strong bear trend!
        if ema_fast > ema_slow:
            score += 20
            reasons.append(f"ADX Strong Trend {adx:.1f} (+20)")
        else:
            if not is_oversold:
                score -= 20
                reasons.append(f"ADX Strong Bear Trend {adx:.1f} (-20)")
            else:
                reasons.append(f"ADX Strong Bear Trend {adx:.1f} (Penalty waived due to Oversold)")
    else:
        # Consolidation bonus: if market is quiet, we can trade mean reversion or range
        score += 15
        reasons.append(f"ADX Low/Consolidation {adx:.1f} (+15)")
    
    # Clamp score
    final_score = max(0.0, min(100.0, score))

    # 4. Determine State & Risk Scaler
    status = "NEUTRAL"
    can_trade = True
    risk_scaler = 0.5
    
    if final_score >= 70:
        status = "BULLISH"
        risk_scaler = 1.0
    elif final_score >= 40:
        status = "NEUTRAL"
        risk_scaler = 0.6
    elif final_score >= 15:
        status = "BEARISH_DEFENSIVE"
        risk_scaler = 0.3
    else:
        status = "CRITICAL"
        can_trade = False
        risk_scaler = 0.0

    reason_str = f"Score {final_score:.0f}/100 (RSI {rsi:.1f}): " + ", ".join(reasons)
    
    # Log logic
    previous_score = regime_last_result.score if regime_last_result.status != "INIT" else -1
    if abs(final_score - previous_score) > 5 or status != regime_last_result.status:
        send_telegram_alert(f"🧭 Regime Update: {status} ({final_score:.0f}/100)\n{reason_str}")

    regime_last_result = RegimeState(
        score=final_score,
        status=status,
        reason=reason_str,
        can_trade=can_trade,
        risk_scaler=risk_scaler
    )
    regime_last_check = now
    
    # Store metrics for logging
    last_regime_metrics = {
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'roc': roc_value,
        'score': final_score,
        'status': status
    }
    
    return regime_last_result


def calculate_position_size(
    candidate: Dict[str, object],
    quote_budget: float,
    risk_scaler: float = 1.0
) -> float:
    """Derive a position size (base units) from a quote notional budget and candidate data."""
    price = candidate.get("price")
    if not isinstance(price, (int, float)) or price <= 0:
        return 0.0
    
    # Scale budget by regime risk
    effective_budget = quote_budget * risk_scaler
    
    exposure_cap_remaining = max(MAX_TOTAL_EXPOSURE_USD - current_total_exposure(), 0.0)
    if exposure_cap_remaining <= 0:
        return 0.0
    trade_notional = min(max(effective_budget, 0.0), exposure_cap_remaining)
    trade_notional = min(trade_notional, exposure_cap_remaining)
    if trade_notional <= 0:
        return 0.0
    return trade_notional / price

# === MAIN LOOP ENTRY POINT ===
def maybe_send_health_heartbeat(regime_state: RegimeState) -> None:
    """Emit a periodic heartbeat to Telegram so stalls are visible quickly."""
    global last_health_heartbeat
    if HEARTBEAT_INTERVAL_SECONDS <= 0:
        return
    now_ts = time.time()
    if last_health_heartbeat and (now_ts - last_health_heartbeat) < HEARTBEAT_INTERVAL_SECONDS:
        return

    regime_label = regime_state.status
    exposure = current_total_exposure()
    message = (
        "⏱️ Omega heartbeat\n"
        f"Loop @ {last_loop_start_str}\n"
        f"Regime: {regime_label} ({regime_state.score:.0f}/100)\n"
        f"Open positions: {len(open_positions)} | Exposure ${exposure:.2f}\n"
        f"Cooldowns tracked: {len(symbol_cooldowns)}\n"
        f"Restricted Symbols: {len(RUNTIME_RESTRICTED_SYMBOLS)}"
    )
    if regime_state.reason:
        message += f"\nReason: {regime_state.reason}"

    send_telegram_alert(message)
    last_health_heartbeat = now_ts


def run_bot():
    global last_loop_start_str, last_portfolio_snapshot, last_regime_warning_ts, last_regime_warning_reason
    log_event("info", "MainLoop", "Starting OMEGA-VX-CRYPTO loop…")
    send_telegram_alert("🚀 OMEGA-VX-CRYPTO bot started loop")
    while True:
        try:
            loop_start = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            last_loop_start_str = loop_start

            regime_state = evaluate_market_regime()
            
            # Use can_trade flag from graded regime logic
            if not regime_state.can_trade:
                if open_positions:
                    monitor_positions()
                now_ts = time.time()
                warn_due = (
                    REGIME_WARN_COOLDOWN_SECONDS == 0
                    or (now_ts - last_regime_warning_ts) >= REGIME_WARN_COOLDOWN_SECONDS
                    or regime_state.status != last_regime_warning_reason
                )
                if warn_due:
                    log_event("warn", "MainLoop", f"Regime Critical: {regime_state.reason}")
                    last_regime_warning_ts = now_ts
                    last_regime_warning_reason = regime_state.status
                maybe_send_health_heartbeat(regime_state)
                time.sleep(config.LOOP_SLEEP_SECONDS)
                continue

            monitor_positions()

            try:
                balance = call_with_retries(exchange.fetch_balance)
                free_balances = balance.get('free') or {}
                quote_asset, quote_available = pick_quote_balance(free_balances)
            except Exception as balance_err:
                log_event("warn", "MainLoop", f"Failed to fetch balance: {balance_err}")
                time.sleep(config.LOOP_SLEEP_SECONDS)
                continue

            quote_budget = max(quote_available * 0.95, 0.0)
            per_trade_cap = quote_available * RISK_PER_TRADE_PCT

            candidates = scan_top_cryptos(
                regime_state=regime_state,
                limit=SCANNER_MAX_CANDIDATES, 
                quote_asset=quote_asset
            )
            if not candidates:
                log_event("info", "MainLoop", "No valid candidates this cycle.")
                time.sleep(config.LOOP_SLEEP_SECONDS)
                continue

            for candidate in candidates:
                symbol = candidate.get("symbol")
                if not symbol or symbol.upper() in RESTRICTED_SYMBOLS:
                    continue
                if symbol in open_positions:
                    continue
                if not validate_entry_conditions(candidate):
                    continue
                if quote_budget <= 0:
                    log_event("warn", "MainLoop", "Quote budget exhausted for this cycle.")
                    break

                trade_notional = min(quote_budget, per_trade_cap)
                position_size = calculate_position_size(
                    candidate, 
                    trade_notional, 
                    risk_scaler=regime_state.risk_scaler
                )
                if position_size <= 0:
                    continue

                order_result = execute_trade(
                    symbol,
                    "buy",
                    position_size,
                    candidate.get("price"),
                    reason="Signal entry"
                )
                if not order_result:
                    continue

                if order_result.get("status") != "dry_run":
                    filled_amt = order_result.get("filled_amount")
                    executed_price = order_result.get("executed_price") or candidate.get("price")
                    log_trade_features(
                        symbol=symbol,
                        score=candidate.get("score"),
                        indicators=candidate.get("indicators"),
                        reasons=candidate.get("reasons"),
                        filled_amount=filled_amt,
                        executed_price=executed_price,
                        actual_notional=(filled_amt or 0) * (executed_price or 0),
                        effective_stop_pct=(
                            ATR_HARD_STOP_MULTIPLIER * float(candidate.get("atr_pct"))
                            if isinstance(candidate.get("atr_pct"), (int, float))
                            else None
                        ),
                    )
                    quote_budget = max(quote_budget - trade_notional, 0.0)

            now_ts = time.time()
            if now_ts - last_portfolio_snapshot > PORTFOLIO_SNAPSHOT_INTERVAL_SECONDS:
                log_portfolio_snapshot()
                last_portfolio_snapshot = now_ts

            maybe_send_health_heartbeat(regime_state)
            time.sleep(config.LOOP_SLEEP_SECONDS)
        except (NetworkError, ExchangeError) as loop_err:
            log_event("warn", "MainLoop", f"Exchange connectivity issue: {loop_err}")
            time.sleep(config.LOOP_SLEEP_SECONDS)
        except Exception as loop_err:
            log_event("error", "MainLoop", f"Unexpected error: {loop_err}")
            time.sleep(config.LOOP_SLEEP_SECONDS)


if __name__ == "__main__":
    if SKIP_BOT_INIT:
        print("OMEGA_SKIP_INIT active; skipping bot runtime.")
    else:
        run_bot()
