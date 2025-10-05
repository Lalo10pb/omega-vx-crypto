import ccxt
import json
import os
import time
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import csv
import os

env_path = os.path.join(os.path.dirname(__file__), ".env")
print(f"📦 Loading environment from: {env_path}")
if load_dotenv(dotenv_path=env_path):
    print("🔑 Environment variables loaded from .env.")
else:
    print("⚠️ .env file not found or could not be loaded; relying on existing environment.")
import numpy as np
import requests
import csv
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

print("🟢 OMEGA-VX-CRYPTO bot started.")
open_positions = set()
open_positions_data = {}
last_buy_time = defaultdict(lambda: 0)
last_trade_time = defaultdict(lambda: 0)
COOLDOWN_SECONDS = 30 * 60  # 30 minutes
TRADE_COOLDOWN_SECONDS = 60 * 60  # 1 hour global cooldown per symbol

markets_cache: Dict[str, dict] = {}
markets_last_load: float = 0.0
ohlcv_cache: Dict[Tuple[str, str, int], Dict[str, object]] = {}

regime_last_check: float = 0.0
regime_last_result: Tuple[bool, str] = (True, "init")

orderbook_cache: Dict[Tuple[str, float], Dict[str, object]] = {}
listing_age_cache: Dict[str, Dict[str, float]] = {}

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", 5))
MAX_TOTAL_EXPOSURE_USD = float(os.getenv("MAX_TOTAL_EXPOSURE_USD", 500.0))
STATE_PATH = os.getenv("BOT_STATE_PATH", "bot_state.json")
LIMIT_PRICE_BUFFER = float(os.getenv("LIMIT_PRICE_BUFFER", 0.001))
MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME_24H", 100_000))
_MIN_QUOTE_VOLUME_FALLBACK = float(
    os.getenv("MIN_QUOTE_VOLUME_FALLBACK", 60_000)
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
last_regime_metrics: Dict[str, float] = {}
last_weekly_summary_date: Optional[str] = None
last_portfolio_snapshot: float = 0.0
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
    for s in os.getenv("RESTRICTED_SYMBOLS", "EUR/USD,EUR/USDT,EUR/USDC").split(",")
    if s.strip()
}
PORTFOLIO_SNAPSHOT_INTERVAL_SECONDS = int(os.getenv("PORTFOLIO_SNAPSHOT_INTERVAL_SECONDS", 1800))

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
        response = requests.post(url, json=payload)
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
    global open_positions, open_positions_data, last_buy_time, last_trade_time
    if not STATE_PATH or not os.path.exists(STATE_PATH):
        return
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as state_file:
            data = json.load(state_file)
        open_positions = set(data.get("open_positions", []))
        open_positions_data = data.get("open_positions_data", {}) or {}
        last_buy = {k: float(v) for k, v in (data.get("last_buy_time") or {}).items()}
        last_trade = {k: float(v) for k, v in (data.get("last_trade_time") or {}).items()}
        last_buy_time = defaultdict(lambda: 0, last_buy)
        last_trade_time = defaultdict(lambda: 0, last_trade)
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
        }
        with open(STATE_PATH, "w", encoding="utf-8") as state_file:
            json.dump(payload, state_file, ensure_ascii=True, indent=2)
    except Exception as err:
        print(f"⚠️ Failed to persist bot state: {err}")


def get_markets(force: bool = False) -> Dict[str, dict]:
    global markets_cache, markets_last_load
    now = time.time()
    if force or not markets_cache or (now - markets_last_load) > MARKET_REFRESH_SECONDS:
        try:
            markets_cache = exchange.load_markets()
            markets_last_load = now
        except Exception as err:
            print(f"⚠️ Failed to refresh markets: {err}")
    return markets_cache


def fetch_ohlcv_cached(symbol: str, timeframe: str, limit: int, ttl: Optional[int] = None) -> List[List[float]]:
    if ttl is None:
        ttl = MULTI_TF_CACHE_TTL_SECONDS if timeframe in {"4h", "1d", "1w"} else OHLCV_CACHE_TTL_SECONDS
    key = (symbol, timeframe, limit)
    now = time.time()
    cached = ohlcv_cache.get(key)
    if cached and now - cached['ts'] < ttl:
        return cached['data']
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    ohlcv_cache[key] = {'ts': now, 'data': data}
    return data


def fetch_orderbook_metrics(symbol: str, depth_percent: float = DEPTH_BAND_PERCENT) -> Dict[str, float]:
    key = (symbol, depth_percent)
    now = time.time()
    cached = orderbook_cache.get(key)
    if cached and (now - cached.get('timestamp', 0.0)) < ORDERBOOK_CACHE_TTL_SECONDS:
        return cached

    try:
        orderbook = exchange.fetch_order_book(symbol, limit=DEPTH_ORDERBOOK_LIMIT)
    except Exception as err:
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


load_bot_state()
get_markets(force=True)


def get_gspread_client():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('google_credentials.json', scope)
        client = gspread.authorize(creds)
        return client
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
            balance = exchange.fetch_balance()
            print("💰 Kraken Balance Keys:", list(balance['total'].keys()))
            usd = balance['total'].get('USD', 0)
            total = sum(v for v in balance['total'].values() if isinstance(v, (int, float)))
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
                ticker = exchange.fetch_ticker(sym)
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

def execute_trade(symbol, side, amount, price=None, reason: str = "Live trade executed") -> Optional[Dict[str, float]]:
    now = time.time()
    if side == "buy":
        if symbol in open_positions:
            reason = f"⛔ Trade rejected: {symbol} is already in open_positions."
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
        print(f"🛒 Placing {side.upper()} order for {symbol} on {exchange_name.upper()}...")
        if amount is None or amount <= 0:
            message = f"⚠️ Invalid trade amount for {symbol}; skipping {side} order."
            print(message)
            send_telegram_alert(message)
            return None
        ticker = exchange.fetch_ticker(symbol)
        reference_price = extract_valid_price(ticker)
        if reference_price is None:
            message = f"⚠️ No valid price data available for {symbol}; skipping {side} order."
            print(message)
            send_telegram_alert(message)
            return None

        try:
            order_book = exchange.fetch_order_book(symbol, limit=10)
        except Exception as book_err:
            message = f"⚠️ Failed to fetch order book for {symbol}: {book_err}"
            print(message)
            send_telegram_alert(message)
            return None

        side_levels = order_book.get('asks' if side == "buy" else 'bids') or []
        side_levels = [level for level in side_levels if isinstance(level, list) and len(level) >= 2]
        if not side_levels:
            message = f"⚠️ Order book empty for {symbol}; skipping {side} order."
            print(message)
            send_telegram_alert(message)
            return None

        book_price = side_levels[0][0]
        if not isinstance(book_price, (int, float)) or book_price <= 0:
            message = f"⚠️ Order book price invalid for {symbol}; skipping {side} order."
            print(message)
            send_telegram_alert(message)
            return None

        available_volume = sum(level[1] for level in side_levels if isinstance(level[1], (int, float)))
        if side == "sell":
            position = open_positions_data.get(symbol, {})
            position_amount = position.get('amount', 0)
            amount = min(amount, position_amount) if position_amount else amount
            if amount <= 0:
                message = f"⚠️ No position size available for {symbol}; skipping sell."
                print(message)
                send_telegram_alert(message)
                return None

        fallback_price = price if isinstance(price, (int, float)) and price > 0 else reference_price
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
            print(message)
            send_telegram_alert(message)
            return None
        if amount <= 0:
            message = f"⚠️ Order size rounded to zero for {symbol}; skipping {side} order."
            print(message)
            send_telegram_alert(message)
            return None

        if available_volume < amount:
            message = (
                f"⚠️ Not enough liquidity to {side} {amount} {symbol}; "
                f"available {available_volume:.4f}."
            )
            print(message)
            send_telegram_alert(message)
            return None

        if side == "buy":
            projected_exposure = current_total_exposure() + (fallback_price * amount)
            if projected_exposure > MAX_TOTAL_EXPOSURE_USD:
                message = (
                    f"🚫 Exposure cap hit: projected {projected_exposure:.2f} exceeds {MAX_TOTAL_EXPOSURE_USD:.2f}; "
                    f"skipping {symbol} buy."
                )
                print(message)
                send_telegram_alert(message)
                return None

        # --- Begin retry block for buy logic ---
        if side == "buy":
            try:
                order = exchange.create_order(symbol, "limit", "buy", amount, limit_price)
            except Exception as e:
                print(f"⚠️ Limit order failed, retrying with market order: {e}")
                try:
                    order = exchange.create_order(symbol, "market", "buy", amount)
                except Exception as e2:
                    print(f"❌ Market order also failed: {e2}")
                    send_telegram_alert(f"❌ Trade failed for {symbol}. Limit and market both failed.")
                    return None
        elif side == "sell":
            order = exchange.create_limit_sell_order(symbol, amount, limit_price)
        else:
            send_telegram_alert(f"❌ Invalid trade side: {side}")
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
            print(message)
            send_telegram_alert(message)
            return None

        executed_price = final_order.get('average') or final_order.get('price') or fallback_price
        if executed_price is None or executed_price <= 0:
            executed_price = fallback_price

        if executed_price is None or executed_price <= 0:
            message = f"⚠️ {symbol} execution price unavailable; skipping trade log."
            print(message)
            send_telegram_alert(message)
            return None

        executed_price = float(executed_price)
        if filled_amount < amount:
            send_telegram_alert(
                f"ℹ️ Partial fill for {symbol} {side}: filled {filled_amount} of {amount}. Remaining order cancelled."
            )

        last_trade_time[symbol] = now
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
    except Exception as e:
        send_telegram_alert(f"❌ Trade execution failed: {e}")
        print(f"❌ Trade execution failed: {e}")
        return None

# === Monitor & Auto-Close Open Positions ===
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

# === Improved Coin Scanner ===
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
    limit: int = 5,
    quote_asset: Optional[str] = None,
    min_volume: Optional[float] = None,
    allow_fallback: bool = True,
) -> List[Dict[str, object]]:
    try:
        quote_asset = (quote_asset or DEFAULT_QUOTE_ASSET).upper()
        print(f"🔍 Scanning {exchange_name.upper()} {quote_asset} markets with enhanced filters...")
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

        restricted_keywords = ["RETARDIO", "SPICE", "KERNEL", "HIPPO", "MERL", "DEGEN", "BMT"]

        tickers = list(markets.items())
        print(f"🔍 Starting scan: {len(tickers)} tokens to evaluate...", flush=True)

        def log_skip(symbol_name: str, reason: str) -> None:
            print(f"🚫 Skipping {symbol_name} — {reason}", flush=True)

        for symbol, market in tickers:
            if symbol.upper() in RESTRICTED_SYMBOLS:
                log_skip(symbol, "restricted symbol")
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
                ticker = exchange.fetch_ticker(symbol)
            except Exception as fetch_err:
                log_skip(symbol, f"ticker fetch failed ({fetch_err})")
                continue

            quote_volume = extract_quote_volume(ticker)
            if quote_volume is None or quote_volume < volume_threshold:
                vol_text = (
                    "unavailable"
                    if quote_volume is None
                    else f"{quote_volume:.2f} < min {volume_threshold:.2f}"
                )
                log_skip(symbol, f"failed volume filter ({vol_text})")
                continue

            bid = ticker.get('bid')
            ask = ticker.get('ask')
            if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)):
                log_skip(symbol, "bid/ask not numeric")
                continue
            if bid <= 0 or ask <= 0 or ask <= bid:
                log_skip(symbol, "invalid bid/ask spread")
                continue
            mid_price = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid_price) * 100 if mid_price > 0 else None
            if spread_pct is None or spread_pct > MAX_SPREAD_PERCENT:
                if spread_pct is None:
                    log_skip(symbol, "spread unavailable")
                else:
                    log_skip(symbol, f"failed spread filter ({spread_pct:.2f}% > {MAX_SPREAD_PERCENT:.2f}%)")
                continue

            depth_metrics = fetch_orderbook_metrics(symbol)
            if not depth_metrics:
                log_skip(symbol, "order book metrics unavailable")
                continue
            depth_spread_pct = depth_metrics.get('spread_pct')
            if isinstance(depth_spread_pct, (int, float)) and depth_spread_pct > MAX_SPREAD_PERCENT:
                log_skip(symbol, f"order book spread high ({depth_spread_pct:.2f}% > {MAX_SPREAD_PERCENT:.2f}%)")
                continue
            depth_ratio = depth_metrics.get('depth_ratio')
            if depth_ratio is None or depth_ratio < MIN_DEPTH_IMBALANCE:
                ratio_text = "unavailable" if depth_ratio is None else f"{depth_ratio:.2f} < {MIN_DEPTH_IMBALANCE:.2f}"
                log_skip(symbol, f"failed depth imbalance ({ratio_text})")
                continue
            depth_mid_price = depth_metrics.get('mid_price') or mid_price
            bid_band_volume = float(depth_metrics.get('bid_volume_band') or 0.0)
            ask_band_volume = float(depth_metrics.get('ask_volume_band') or 0.0)
            bid_band_notional = bid_band_volume * depth_mid_price
            ask_band_notional = ask_band_volume * depth_mid_price
            if bid_band_notional < MIN_DEPTH_NOTIONAL_USD or ask_band_notional < MIN_DEPTH_NOTIONAL_USD:
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

            score = 0
            reasons: List[str] = []

            if ema9 > ema21 > ema50:
                score += 2
                reasons.append("1h EMA stack bullish")
            if fresh_signal:
                score += 1
                reasons.append(f"Fresh EMA cross ({fresh_signal_age} bars ago)")
            if 45 <= rsi_1h <= 75:
                score += 1
                reasons.append(f"RSI in momentum zone ({rsi_1h:.1f})")
            if volume_ratio and volume_ratio >= 1.5:
                score += 1
                reasons.append(f"Volume x{volume_ratio:.2f} vs avg")
            if adx_1h and adx_1h >= 20:
                score += 1
                reasons.append(f"ADX strong ({adx_1h:.1f})")
            if macd_hist_slope and macd_hist_slope > 0:
                score += 1
                reasons.append("MACD momentum rising")
            if roc_pct and roc_pct > 0.3:
                score += 1
                reasons.append(f"ROC positive ({roc_pct:.2f}%)")
            if timeframe_alignment_flags:
                score += len(timeframe_alignment_flags)
                reasons.append("Multi-timeframe uptrend: " + ", ".join(timeframe_alignment_flags))
            if depth_ratio and depth_ratio >= MIN_DEPTH_IMBALANCE:
                score += 1
                reasons.append(
                    f"Order-book bids {depth_ratio:.2f}x asks inside {DEPTH_BAND_PERCENT:.1f}%"
                )
            if is_new_listing:
                score += 1
                reasons.append(f"Fresh Kraken listing (~{listing_age_days:.1f}d)")

            # Compute extra diagnostics for logging
            ema_slope = (ema9 - ema21) / ema21 if ema21 else 0.0
            volatility = atr_pct or 0.0
            print(
                f"🔬 {symbol} — Score: {score} | EMA Slope: {ema_slope:.4f} | "
                f"RSI: {rsi_1h:.2f} | Volatility: {volatility:.4f}",
                flush=True
            )

            if score < 2:
                print(f"⏭️ Skipped {symbol}: Did not pass filters (score={score})", flush=True)
                continue

            candidate = {
                "symbol": symbol,
                "score": score,
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
                )
            return []

        candidates.sort(key=lambda item: (-(item.get('score') or 0), -(item.get('indicators', {}).get('quote_volume') or 0)))
        top_candidates = candidates[:limit]
        log_scanner_snapshot(top_candidates)
        print("✅ Scanner picks:", [(c['symbol'], c['score']) for c in top_candidates])
        return top_candidates
    except Exception as e:
        send_telegram_alert(f"❌ Enhanced scan error: {e}")
        return []


def validate_entry_conditions(symbol: str) -> Tuple[bool, str, Dict[str, float]]:
    try:
        ticker = exchange.fetch_ticker(symbol)
    except Exception as err:
        return False, f"Ticker fetch failed: {err}", {}

    quote_volume = extract_quote_volume(ticker)
    if quote_volume is None or quote_volume < MIN_QUOTE_VOLUME_24H:
        return False, "Quote volume below minimum threshold", {}

    bid = ticker.get('bid')
    ask = ticker.get('ask')
    if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)) or bid <= 0 or ask <= 0 or ask <= bid:
        return False, "Spread check failed", {}
    mid_price = (bid + ask) / 2
    spread_pct = ((ask - bid) / mid_price) * 100 if mid_price > 0 else None
    if spread_pct is None or spread_pct > MAX_SPREAD_PERCENT:
        return False, "Spread too wide", {}

    try:
        ohlcv = fetch_ohlcv_cached(symbol, timeframe='1h', limit=150, ttl=OHLCV_CACHE_TTL_SECONDS)
    except Exception as err:
        return False, f"OHLCV fetch failed: {err}", {}

    if not ohlcv or len(ohlcv) < 80:
        return False, "Insufficient OHLCV data", {}

    closes = [c[4] for c in ohlcv]
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    volumes = [c[5] for c in ohlcv]

    if not closes or closes[-1] is None:
        return False, "Invalid close data", {}

    last_price = float(closes[-1])
    if last_price <= 0:
        return False, "Last price invalid", {}

    ema9_series = calculate_ema_series(closes, 9)
    ema21_series = calculate_ema_series(closes, 21)
    ema50_series = calculate_ema_series(closes, 50)
    if not ema9_series or not ema21_series or not ema50_series:
        return False, "EMA series unavailable", {}

    ema9 = ema9_series[-1]
    ema21 = ema21_series[-1]
    ema50 = ema50_series[-1]
    if not (ema9 > ema21 > ema50):
        return False, "EMA stack lost", {}

    rsi_values = calculate_rsi(closes)
    if not rsi_values:
        return False, "RSI unavailable", {}
    rsi = rsi_values[-1]
    if rsi < 40 or rsi > 78:
        return False, f"RSI out of range ({rsi:.1f})", {}

    atr_value = compute_atr(highs, lows, closes)
    atr_pct = (atr_value / last_price) if atr_value else None
    if atr_pct and atr_pct > MAX_ATR_PERCENT:
        return False, f"ATR too high ({atr_pct:.3f})", {}

    avg_vol_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else None
    volume_ratio = (volumes[-1] / avg_vol_20) if avg_vol_20 else None
    if volume_ratio and volume_ratio < 1.0:
        return False, "Volume momentum faded", {}

    return True, "", {
        "last_price": last_price,
        "quote_volume": quote_volume,
        "spread_pct": spread_pct,
        "rsi": rsi,
        "atr_pct": atr_pct,
        "volume_ratio": volume_ratio,
    }


def evaluate_market_regime(force: bool = False) -> Tuple[bool, str]:
    global regime_last_check, regime_last_result, last_regime_metrics, last_loop_start_str
    now = time.time()
    if not force and (now - regime_last_check) < REGIME_CACHE_SECONDS and regime_last_result is not None:
        return regime_last_result

    minimum_required = max(REGIME_LOOKBACK // 2, REGIME_EMA_SLOW * 3)
    try:
        ohlcv = fetch_ohlcv_cached(
            REGIME_SYMBOL,
            timeframe=REGIME_TIMEFRAME,
            limit=max(REGIME_LOOKBACK, minimum_required),
            ttl=REGIME_CACHE_SECONDS,
        )
    except Exception as err:
        reason = f"Regime check failed: {err}"
        regime_last_result = (False, reason)
        regime_last_check = now
        return regime_last_result

    if not ohlcv or len(ohlcv) < minimum_required:
        reason = "Insufficient regime OHLCV data"
        regime_last_result = (False, reason)
        regime_last_check = now
        return regime_last_result

    closes = [c[4] for c in ohlcv if isinstance(c[4], (int, float)) and c[4] > 0]
    if len(closes) < minimum_required:
        reason = "Regime close data invalid"
        regime_last_result = (False, reason)
        regime_last_check = now
        return regime_last_result

    ema_fast_series = calculate_ema_series(closes, REGIME_EMA_FAST)
    ema_slow_series = calculate_ema_series(closes, REGIME_EMA_SLOW)
    if not ema_fast_series or not ema_slow_series:
        reason = "Regime EMA data missing"
        regime_last_result = (False, reason)
        regime_last_check = now
        return regime_last_result

    ema_fast = ema_fast_series[-1]
    ema_slow = ema_slow_series[-1]
    roc_period = min(max(REGIME_EMA_SLOW, 10), len(closes) - 2)
    roc_value = rate_of_change(closes, period=roc_period)
    if roc_value is None:
        roc_value = 0.0

    last_regime_metrics = {
        'ema_fast': float(ema_fast),
        'ema_slow': float(ema_slow),
        'roc': float(roc_value),
    }

    bullish = ema_fast > ema_slow and roc_value >= REGIME_MIN_ROC
    if bullish:
        reason = (
            f"EMA{REGIME_EMA_FAST}>{REGIME_EMA_SLOW} ({ema_fast:.2f}>{ema_slow:.2f}) "
            f"and ROC {roc_value:.2f}%"
    )
    else:
        condition = "EMA stack" if ema_fast <= ema_slow else "ROC"
        # Custom regime defensive log and alert
        loop_label = last_loop_start_str or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        ema_21 = float(ema_fast)
        ema_55 = float(ema_slow)
        roc = float(roc_value)
        print(
            f"🛡️ Regime defensive at {loop_label}: EMA21={ema_21:.2f}, "
            f"EMA55={ema_55:.2f}, ROC={roc:.2f}%",
            flush=True
        )
        global last_telegram_notify
        if datetime.now(timezone.utc) - last_telegram_notify > timedelta(hours=3):
            send_telegram_alert(f"⚠️ Omega-Crypto standing down: Weak trend regime (EMA21={ema_fast:.2f}, EMA55={ema_slow:.2f}, ROC={roc_value:.2f}%).")
            last_telegram_notify = datetime.now(timezone.utc)
        reason = (
            f"Regime defensive: {condition} check failed "
            f"(EMA{REGIME_EMA_FAST}={ema_fast:.2f}, EMA{REGIME_EMA_SLOW}={ema_slow:.2f}, ROC={roc_value:.2f}%)"
        )

    previous_state = regime_last_result[0] if regime_last_result else None
    regime_last_result = (bullish, reason)
    regime_last_check = now

    if previous_state is None or previous_state != bullish:
        status = "BULLISH" if bullish else "DEFENSIVE"
        send_telegram_alert(f"🧭 Market regime -> {status}: {reason}")

    return regime_last_result


def allocate_trade_sizes(candidates: List[Dict[str, object]], total_allocatable: float) -> Dict[str, float]:
    if total_allocatable <= 0 or not candidates:
        return {}
    weights = []
    for candidate in candidates:
        indicators = candidate.get('indicators', {})
        atr_pct = indicators.get('atr_pct') or 0.02
        atr_pct = max(float(atr_pct), 0.01)
        weights.append(1 / atr_pct)
    weight_sum = sum(weights)
    allocations: Dict[str, float] = {}
    if weight_sum <= 0:
        per_symbol = total_allocatable / len(candidates)
        for candidate in candidates:
            allocations[candidate['symbol']] = per_symbol
        return allocations
    for candidate, weight in zip(candidates, weights):
        allocations[candidate['symbol']] = total_allocatable * (weight / weight_sum)
    return allocations

# === Main Bot Loop ===
def run_bot():
    global last_portfolio_snapshot
    print("🔁 Starting OMEGA-VX-CRYPTO bot loop...")
    send_telegram_alert("🚀 OMEGA-VX-CRYPTO bot started loop")
    while True:
        print("🌀 Cycle started: scanning market and monitoring positions...")
        loop_start = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        global last_loop_start_str
        last_loop_start_str = loop_start
        print(f"🔁 Loop started at {loop_start} UTC", flush=True)
        try:
            monitor_positions()
            regime_ok, regime_reason = evaluate_market_regime()
            if not regime_ok:
                metrics = last_regime_metrics or {}
                ema_21 = metrics.get('ema_fast')
                ema_55 = metrics.get('ema_slow')
                roc = metrics.get('roc')
                if all(isinstance(val, (int, float)) for val in (ema_21, ema_55, roc)):
                    print(
                        f"🛡️ Regime defensive at {loop_start}: EMA21={ema_21:.2f}, "
                        f"EMA55={ema_55:.2f}, ROC={roc:.2f}%",
                        flush=True
                    )
                else:
                    print(f"🛡️ Regime defensive at {loop_start}: {regime_reason}", flush=True)
                print("📉 Market conditions not favorable — skipping trade evaluation this cycle.")
                time.sleep(300)
                continue

            # Adaptive trade amount based on available quote balance
            try:
                balance = exchange.fetch_balance()
                free_balances = balance.get('free') or {}
                quote_asset, quote_available = pick_quote_balance(free_balances)
                if quote_available < 1:
                    print(f"⚠️ Not enough free {quote_asset} balance to trade.")
                    time.sleep(10)
                    continue
            except Exception as e:
                send_telegram_alert(f"⚠️ Failed to fetch balance for dynamic trade sizing: {e}")
                time.sleep(10)
                continue

            candidates = scan_top_cryptos(limit=SCANNER_MAX_CANDIDATES, quote_asset=quote_asset)
            if not candidates:
                print("⚠️ No valid pairs found.")
                time.sleep(10)
                continue

            total_allocatable = quote_available * 0.95
            allocations = allocate_trade_sizes(candidates, total_allocatable)
            risk_budget_remaining = max(quote_available * RISK_PER_TRADE_PCT, 0.0)
            print(
                f"💰 Total {quote_asset}: {quote_available:.2f}, Allocatable: {total_allocatable:.2f}, "
                f"Candidates: {[(c['symbol'], round(allocations.get(c['symbol'], 0), 2)) for c in candidates]}"
            )

            summary = ", ".join([f"{c['symbol']} (score {c['score']})" for c in candidates])
            send_telegram_alert(f"🧠 Scanner picks: {summary}")

            for candidate in candidates:
                symbol = candidate[0] if isinstance(candidate, (list, tuple)) else candidate.get('symbol')
                if isinstance(candidate, (list, tuple)):
                    # If candidate is a tuple/list, try to build a dict-like interface for compatibility
                    candidate_dict = {}
                    if len(candidate) > 0:
                        candidate_dict['symbol'] = candidate[0]
                    if len(candidate) > 1:
                        candidate_dict['score'] = candidate[1]
                    candidate = candidate_dict
                if symbol.upper() in RESTRICTED_SYMBOLS:
                    print(f"🚫 {symbol} entry blocked: symbol restricted.")
                    continue
                quote_budget = allocations.get(symbol, 0)
                if quote_budget <= 0:
                    print(f"⚠️ No {quote_asset} allocation for {symbol}; skipping.")
                    continue
                if symbol in open_positions:
                    print(f"⏸️ Already holding {symbol}; skipping new entry.")
                    continue
                if risk_budget_remaining <= 0:
                    print("⚠️ Risk budget exhausted; skipping remaining candidates.")
                    break

                valid, rejection_reason, metrics = validate_entry_conditions(symbol)
                if not valid:
                    print(f"🚫 {symbol} entry blocked: {rejection_reason}")
                    continue

                price = metrics.get('last_price') or candidate.get('indicators', {}).get('last_price') if isinstance(candidate, dict) else None
                if not price or price <= 0:
                    print(f"⚠️ Unable to determine price for {symbol}; skipping.")
                    continue

                atr_pct = metrics.get('atr_pct')
                atr_stop_pct = None
                if isinstance(atr_pct, (int, float)) and atr_pct > 0:
                    atr_stop_pct = ATR_HARD_STOP_MULTIPLIER * float(atr_pct)
                effective_stop_pct = HARD_STOP_PERCENT / 100
                if atr_stop_pct and atr_stop_pct > 0:
                    effective_stop_pct = min(effective_stop_pct, atr_stop_pct)
                effective_stop_pct = max(effective_stop_pct, 0.0001)

                max_position_from_risk = risk_budget_remaining / effective_stop_pct
                if max_position_from_risk <= 0:
                    print("⚠️ No remaining risk capacity; stopping entries.")
                    break

                quote_budget = min(quote_budget, max_position_from_risk)
                amount = quote_budget / price
                if amount <= 0:
                    print(f"⚠️ Computed trade amount invalid for {symbol}; skipping.")
                    continue

                min_cost, min_amount = fetch_market_minimums(symbol)
                if min_cost and quote_budget < min_cost:
                    print(
                        f"🚫 {symbol} entry blocked: budget ${quote_budget:.2f} below Kraken minimum "
                        f"${min_cost:.2f}."
                    )
                    continue
                if min_amount and amount < min_amount:
                    required_notional = min_amount * price
                    print(
                        f"🚫 {symbol} entry blocked: amount {amount:.6f} < min {min_amount:.6f} "
                        f"(~${required_notional:.2f})."
                    )
                    continue

                spread_value = metrics.get('spread_pct')
                spread_text = f"{spread_value:.3f}%" if isinstance(spread_value, (int, float)) else "N/A"
                entry_reason = "Scanner entry"
                reason_components = candidate.get('reasons') if isinstance(candidate, dict) else None
                if reason_components:
                    entry_reason = "Scanner entry: " + "; ".join(reason_components)

                combined_indicators = dict(candidate.get('indicators') or {}) if isinstance(candidate, dict) else {}
                for key, value in (metrics or {}).items():
                    if value is not None:
                        combined_indicators[key] = value

                print(
                    f"✅ Executing candidate {symbol}: {quote_asset} {quote_budget:.2f}, amount {amount:.6f}, "
                    f"spread {spread_text}, stop risk {effective_stop_pct * 100:.2f}%"
                )
                trade_result = execute_trade(symbol, "buy", amount, reason=entry_reason)
                if trade_result:
                    executed_amount = float(trade_result.get('filled_amount') or 0.0)
                    executed_price = float(trade_result.get('executed_price') or 0.0)
                    if executed_amount > 0 and executed_price > 0:
                        actual_notional = executed_price * executed_amount
                        trade_risk_used = actual_notional * effective_stop_pct
                        log_trade_features(
                            symbol,
                            candidate.get('score') if isinstance(candidate, dict) else None,
                            combined_indicators,
                            reason_components,
                            executed_amount,
                            executed_price,
                            actual_notional=actual_notional,
                            effective_stop_pct=effective_stop_pct,
                        )
                        risk_budget_remaining = max(risk_budget_remaining - trade_risk_used, 0.0)
                        print(
                            f"🧮 Risk budget remaining: ${risk_budget_remaining:.2f} "
                            f"(consumed ${trade_risk_used:.2f} at {effective_stop_pct * 100:.2f}% stop)"
                        )
                        if risk_budget_remaining <= 0:
                            print("✅ Risk budget consumed for this cycle; halting additional entries.")
                            break
                    else:
                        print("⚠️ Trade fill data incomplete; skipping feature log and risk update.")

            try:
                now_ts = time.time()
                if now_ts - last_portfolio_snapshot >= PORTFOLIO_SNAPSHOT_INTERVAL_SECONDS:
                    log_portfolio_snapshot()
                    last_portfolio_snapshot = now_ts
            except Exception as snapshot_err:
                print(f"⚠️ Portfolio snapshot logging failed: {snapshot_err}")

            # At the end of the loop, sleep 30 seconds for normal bullish cycles
            time.sleep(30)
        except Exception as cycle_err:
            message = f"❌ Bot cycle error: {cycle_err}"
            print(message)
            send_telegram_alert(message)
            time.sleep(60)


if __name__ == "__main__":
    run_bot()
