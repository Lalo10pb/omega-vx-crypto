import ccxt
import json
import os
import time
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv

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

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", 5))
MAX_TOTAL_EXPOSURE_USD = float(os.getenv("MAX_TOTAL_EXPOSURE_USD", 500.0))
STATE_PATH = os.getenv("BOT_STATE_PATH", "bot_state.json")
LIMIT_PRICE_BUFFER = float(os.getenv("LIMIT_PRICE_BUFFER", 0.001))
MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME_24H", 250_000))
MAX_SPREAD_PERCENT = float(os.getenv("MAX_SPREAD_PERCENT", 0.35))  # expressed in percent
MAX_ATR_PERCENT = float(os.getenv("MAX_ATR_PERCENT", 0.07))
FRESH_SIGNAL_LOOKBACK = int(os.getenv("FRESH_SIGNAL_LOOKBACK", 3))
SCANNER_LOG_PATH = os.getenv("SCANNER_LOG_PATH", "scanner_evaluations.csv")

 # === CONFIGURATION ===
# trade_amount_usd = 25  # USD amount per trade (adjusted to increase capital usage)

# === Paths ===
TRADE_LOG_PATH = "crypto_trade_log.csv"
TRADE_LOG_PATH_BACKUP = "crypto_trade_log_backup.csv"
PORTFOLIO_LOG_PATH = "crypto_portfolio_log.csv"

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

    plus_di = np.where(smoothed_tr == 0, 0, 100 * (smoothed_plus / smoothed_tr))
    minus_di = np.where(smoothed_tr == 0, 0, 100 * (smoothed_minus / smoothed_tr))
    dx = np.where(
        (plus_di + minus_di) == 0,
        0,
        100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    )

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


def current_total_exposure() -> float:
    exposure = 0.0
    for info in open_positions_data.values():
        price = info.get('entry_price')
        amount = info.get('amount')
        if isinstance(price, (int, float)) and isinstance(amount, (int, float)):
            exposure += max(price, 0) * max(amount, 0)
    return float(exposure)


load_bot_state()


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

    send_telegram_alert(f"📒 LOGGED TRADE: {side.upper()} {symbol} | Amount: {amount} @ ${price:.2f} ({reason})")

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

        # --- Daily PnL summary logic ---
        def summarize_daily_pnl(equity_log_file="logs/equity_log.csv"):
            try:
                df = pd.read_csv(equity_log_file)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                today = datetime.now().date()
                daily_logs = df[df["timestamp"].dt.date == today]
                if len(daily_logs) >= 2:
                    start_value = daily_logs.iloc[0]["portfolio_value"]
                    end_value = daily_logs.iloc[-1]["portfolio_value"]
                    change = end_value - start_value
                    pct_change = (change / start_value) * 100
                    return f"📈 Daily P&L: ${change:.2f} ({pct_change:.2f}%)"
                else:
                    return "📈 Daily P&L: Not enough data for summary yet."
            except Exception as e:
                return f"⚠️ Error generating P&L summary: {e}"
        send_telegram_alert(summarize_daily_pnl())

        print(f"💾 Snapshot: USD=${usd:.2f}, Total=${total:.2f}")
    except Exception as e:
        send_telegram_alert(f"⚠️ Failed to log portfolio snapshot: {str(e)}")

def execute_trade(symbol, side, amount, price=None, reason: str = "Live trade executed"):
    now = time.time()
    if side == "buy":
        if symbol in open_positions:
            reason = f"⛔ Trade rejected: {symbol} is already in open_positions."
            print(reason)
            send_telegram_alert(reason)
            return
        if now - last_buy_time[symbol] < COOLDOWN_SECONDS:
            wait_time = int(COOLDOWN_SECONDS - (now - last_buy_time[symbol])) // 60
            reason = f"⏳ Trade rejected: cooldown active for {symbol} ({wait_time} min remaining)."
            print(reason)
            send_telegram_alert(reason)
            return
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            reason = f"🚫 Max open positions reached ({MAX_OPEN_POSITIONS}); skipping {symbol} buy."
            print(reason)
            send_telegram_alert(reason)
            return
    if side == "buy" and now - last_trade_time[symbol] < TRADE_COOLDOWN_SECONDS:
        wait_min = int((TRADE_COOLDOWN_SECONDS - (now - last_trade_time[symbol])) / 60)
        reason = f"⏳ GLOBAL COOLDOWN: {symbol} trade blocked ({wait_min} min left)."
        print(reason)
        send_telegram_alert(reason)
        return

    try:
        print(f"🛒 Placing {side.upper()} order for {symbol} on {exchange_name.upper()}...")
        if amount is None or amount <= 0:
            message = f"⚠️ Invalid trade amount for {symbol}; skipping {side} order."
            print(message)
            send_telegram_alert(message)
            return
        ticker = exchange.fetch_ticker(symbol)
        reference_price = extract_valid_price(ticker)
        if reference_price is None:
            message = f"⚠️ No valid price data available for {symbol}; skipping {side} order."
            print(message)
            send_telegram_alert(message)
            return

        try:
            order_book = exchange.fetch_order_book(symbol, limit=10)
        except Exception as book_err:
            message = f"⚠️ Failed to fetch order book for {symbol}: {book_err}"
            print(message)
            send_telegram_alert(message)
            return

        side_levels = order_book.get('asks' if side == "buy" else 'bids') or []
        side_levels = [level for level in side_levels if isinstance(level, list) and len(level) >= 2]
        if not side_levels:
            message = f"⚠️ Order book empty for {symbol}; skipping {side} order."
            print(message)
            send_telegram_alert(message)
            return

        book_price = side_levels[0][0]
        if not isinstance(book_price, (int, float)) or book_price <= 0:
            message = f"⚠️ Order book price invalid for {symbol}; skipping {side} order."
            print(message)
            send_telegram_alert(message)
            return

        available_volume = sum(level[1] for level in side_levels if isinstance(level[1], (int, float)))
        if side == "buy" and available_volume < amount:
            message = f"⚠️ Not enough liquidity to buy {amount} {symbol}; available {available_volume:.4f}."
            print(message)
            send_telegram_alert(message)
            return
        if side == "sell":
            position = open_positions_data.get(symbol, {})
            position_amount = position.get('amount', 0)
            amount = min(amount, position_amount) if position_amount else amount
            if amount <= 0:
                message = f"⚠️ No position size available for {symbol}; skipping sell."
                print(message)
                send_telegram_alert(message)
                return
            if available_volume < amount:
                message = f"⚠️ Not enough liquidity to sell {amount} {symbol}; available {available_volume:.4f}."
                print(message)
                send_telegram_alert(message)
                return

        fallback_price = price if isinstance(price, (int, float)) and price > 0 else reference_price
        if LIMIT_PRICE_BUFFER < 0:
            buffer = 0.0
        else:
            buffer = LIMIT_PRICE_BUFFER
        if side == "buy":
            limit_price = book_price * (1 + buffer)
        else:
            limit_price = book_price * (1 - buffer)

        limit_price = round(limit_price, 8)
        if limit_price <= 0:
            message = f"⚠️ Computed limit price invalid for {symbol}; skipping {side} order."
            print(message)
            send_telegram_alert(message)
            return

        if side == "buy":
            projected_exposure = current_total_exposure() + (fallback_price * amount)
            if projected_exposure > MAX_TOTAL_EXPOSURE_USD:
                message = (
                    f"🚫 Exposure cap hit: projected ${projected_exposure:.2f} exceeds ${MAX_TOTAL_EXPOSURE_USD:.2f}; "
                    f"skipping {symbol} buy."
                )
                print(message)
                send_telegram_alert(message)
                return

        if side == "buy":
            order = exchange.create_limit_buy_order(symbol, amount, limit_price)
        elif side == "sell":
            order = exchange.create_limit_sell_order(symbol, amount, limit_price)
        else:
            send_telegram_alert(f"❌ Invalid trade side: {side}")
            return

        print(f"📍LIMIT {side.upper()} for {symbol} @ ${limit_price} (amount: {amount}) placed; waiting for fill...")
        final_order = wait_for_order_fill(symbol, order)
        status = final_order.get('status')
        filled_amount = float(final_order.get('filled') or 0.0)

        if filled_amount <= 0:
            message = f"⏹️ {symbol} {side} order not filled (status: {status})."
            print(message)
            send_telegram_alert(message)
            return

        executed_price = final_order.get('average') or final_order.get('price') or fallback_price
        if executed_price is None or executed_price <= 0:
            executed_price = fallback_price

        if executed_price is None or executed_price <= 0:
            message = f"⚠️ {symbol} execution price unavailable; skipping trade log."
            print(message)
            send_telegram_alert(message)
            return

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
                "amount": filled_amount
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
    except Exception as e:
        send_telegram_alert(f"❌ Trade execution failed: {e}")
        print(f"❌ Trade execution failed: {e}")

# === SELL LOGIC CONFIG ===
trailing_stop_pct = 4.5  # widened to reduce false exits from micro swings
take_profit_pct = 6.0    # increased target to capture stronger breakouts
hard_stop_pct = 4.0      # allows slightly more downside to avoid noise

# === Monitor & Auto-Close Open Positions ===
def monitor_positions():
    print("🔍 Monitoring open positions...")
    for symbol in list(open_positions):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=50)
            if not ohlcv or len(ohlcv) < 2:
                print(f"⚠️ Insufficient OHLCV data for {symbol}; skipping monitoring cycle.")
                continue
            closes = [candle[4] for candle in ohlcv]
            highs = [candle[2] for candle in ohlcv]
            lows = [candle[3] for candle in ohlcv]

            # === ATR calculation ===
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

            # === ATR-based exits ===
            tp_level = entry_price + (3.5 * atr)
            ts_level = entry_price - (2.0 * atr)
            hs_level = entry_price - (2.5 * atr)

            if current_price >= tp_level:
                send_telegram_alert(f"🎯 ATR TAKE PROFIT HIT for {symbol} (+{change_pct:.2f}%)")
                execute_trade(symbol, "sell", amount, current_price, reason="ATR take profit")
            elif current_price <= hs_level:
                send_telegram_alert(f"🛑 ATR HARD STOP triggered for {symbol} ({change_pct:.2f}%)")
                execute_trade(symbol, "sell", amount, current_price, reason="ATR hard stop")
            elif current_price <= ts_level:
                send_telegram_alert(f"📉 ATR TRAILING STOP triggered for {symbol} ({change_pct:.2f}%)")
                execute_trade(symbol, "sell", amount, current_price, reason="ATR trailing stop")
            else:
                print(f"⏳ {symbol} holding: {change_pct:.2f}%")
        except Exception as e:
            send_telegram_alert(f"⚠️ monitor_positions error for {symbol}: {str(e)}")

# === Improved Coin Scanner ===
def log_scanner_snapshot(records: List[Dict]) -> None:
    if not records:
        return
    try:
        timestamp = datetime.now().isoformat()
        file_exists = os.path.exists(SCANNER_LOG_PATH)
        with open(SCANNER_LOG_PATH, mode='a', newline='') as log_file:
            writer = csv.writer(log_file)
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "symbol",
                    "score",
                    "quote_volume",
                    "spread_pct",
                    "atr_pct",
                    "rsi_1h",
                    "adx_1h",
                    "volume_ratio",
                    "macd_hist_slope",
                    "roc_pct",
                    "multi_timeframe_alignment",
                    "fresh_signal_age",
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
                    indicators.get('atr_pct'),
                    indicators.get('rsi_1h'),
                    indicators.get('adx_1h'),
                    indicators.get('volume_ratio'),
                    indicators.get('macd_hist_slope'),
                    indicators.get('roc_pct'),
                    indicators.get('timeframe_alignment'),
                    indicators.get('fresh_signal_age'),
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


def scan_top_cryptos(limit: int = 5) -> List[Dict[str, object]]:
    try:
        print(f"🔍 Scanning {exchange_name.upper()} markets with enhanced filters...")
        markets = exchange.load_markets()
        candidates: List[Dict[str, object]] = []

        if isinstance(exchange, ccxt.kraken):
            quote_currency = "/USD"
        else:
            quote_currency = "/USDT"

        restricted_keywords = ["RETARDIO", "SPICE", "KERNEL", "HIPPO", "MERL", "DEGEN", "BMT"]

        for symbol, market in markets.items():
            if not symbol.endswith(quote_currency):
                continue
            if any(keyword in symbol for keyword in restricted_keywords):
                continue
            if market and not market.get('active', True):
                continue

            try:
                ticker = exchange.fetch_ticker(symbol)
            except Exception:
                continue

            quote_volume = extract_quote_volume(ticker)
            if quote_volume is None or quote_volume < MIN_QUOTE_VOLUME_24H:
                continue

            bid = ticker.get('bid')
            ask = ticker.get('ask')
            if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)):
                continue
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue
            mid_price = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid_price) * 100 if mid_price > 0 else None
            if spread_pct is None or spread_pct > MAX_SPREAD_PERCENT:
                continue

            try:
                ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
            except Exception:
                continue
            if not ohlcv_1h or len(ohlcv_1h) < 100:
                continue

            closes_1h = [c[4] for c in ohlcv_1h]
            highs_1h = [c[2] for c in ohlcv_1h]
            lows_1h = [c[3] for c in ohlcv_1h]
            volumes_1h = [c[5] for c in ohlcv_1h]

            if not closes_1h or closes_1h[-1] is None:
                continue
            last_price = float(closes_1h[-1])
            if last_price <= 0:
                continue

            ema9_series = calculate_ema_series(closes_1h, 9)
            ema21_series = calculate_ema_series(closes_1h, 21)
            ema50_series = calculate_ema_series(closes_1h, 50)
            if not ema9_series or not ema21_series or not ema50_series:
                continue
            ema9 = ema9_series[-1]
            ema21 = ema21_series[-1]
            ema50 = ema50_series[-1]

            rsi_values = calculate_rsi(closes_1h)
            if not rsi_values:
                continue
            rsi_1h = rsi_values[-1]

            adx_values = calculate_adx(highs_1h, lows_1h, closes_1h)
            adx_1h = adx_values[-1] if adx_values else None

            atr_value = compute_atr(highs_1h, lows_1h, closes_1h)
            atr_pct = (atr_value / last_price) if atr_value else None
            if atr_pct and atr_pct > MAX_ATR_PERCENT:
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
                ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=120)
            except Exception:
                ohlcv_4h = []
            try:
                ohlcv_1d = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=120)
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

            if score < 4:
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
                    "fresh_signal_age": fresh_signal_age,
                    "timeframe_alignment": ",".join(timeframe_alignment_flags) if timeframe_alignment_flags else "",
                }
            }
            candidates.append(candidate)

        if not candidates:
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
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=150)
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
    print("🔁 Starting OMEGA-VX-CRYPTO bot loop...")
    send_telegram_alert("🚀 OMEGA-VX-CRYPTO bot started loop")
    while True:
        try:
            # Adaptive trade amount based on available USD balance
            try:
                balance = exchange.fetch_balance()
                free_balances = balance.get('free') or {}
                usd_available = free_balances.get('USD')
                if usd_available is None:
                    usd_available = free_balances.get('ZUSD', 0)
                usd_available = float(usd_available or 0)
                if usd_available < 1:
                    print("⚠️ Not enough free USD balance to trade.")
                    time.sleep(10)
                    continue
            except Exception as e:
                send_telegram_alert(f"⚠️ Failed to fetch balance for dynamic trade sizing: {e}")
                time.sleep(10)
                continue

            candidates = scan_top_cryptos()
            if not candidates:
                print("⚠️ No valid pairs found.")
                time.sleep(10)
                continue

            total_allocatable = usd_available * 0.95
            allocations = allocate_trade_sizes(candidates, total_allocatable)
            print(
                f"💰 Total USD: ${usd_available:.2f}, Allocatable: ${total_allocatable:.2f}, "
                f"Candidates: {[(c['symbol'], round(allocations.get(c['symbol'], 0), 2)) for c in candidates]}"
            )

            summary = ", ".join([f"{c['symbol']} (score {c['score']})" for c in candidates])
            send_telegram_alert(f"🧠 Scanner picks: {summary}")

            for candidate in candidates:
                symbol = candidate['symbol']
                usd_budget = allocations.get(symbol, 0)
                if usd_budget <= 0:
                    print(f"⚠️ No USD allocation for {symbol}; skipping.")
                    continue
                if symbol in open_positions:
                    print(f"⏸️ Already holding {symbol}; skipping new entry.")
                    continue

                valid, rejection_reason, metrics = validate_entry_conditions(symbol)
                if not valid:
                    print(f"🚫 {symbol} entry blocked: {rejection_reason}")
                    continue

                price = metrics.get('last_price') or candidate.get('indicators', {}).get('last_price')
                if not price or price <= 0:
                    print(f"⚠️ Unable to determine price for {symbol}; skipping.")
                    continue

                amount = round(usd_budget / price, 6)
                if amount <= 0:
                    print(f"⚠️ Computed trade amount invalid for {symbol}; skipping.")
                    continue

                spread_value = metrics.get('spread_pct')
                spread_text = f"{spread_value:.3f}%" if isinstance(spread_value, (int, float)) else "N/A"
                entry_reason = "Scanner entry"
                reason_components = candidate.get('reasons')
                if reason_components:
                    entry_reason = "Scanner entry: " + "; ".join(reason_components)

                print(
                    f"✅ Executing candidate {symbol}: USD ${usd_budget:.2f}, amount {amount}, "
                    f"spread {spread_text}"
                )
                execute_trade(symbol, "buy", amount, reason=entry_reason)

            monitor_positions()
            log_portfolio_snapshot()
            now_dt = datetime.now()
            if now_dt.weekday() == 6 and now_dt.hour == 17 and now_dt.minute < 5:
                summarize_weekly_pnl()
            print("📌 Current open positions:", open_positions)
            time.sleep(30)
        except Exception as e:
            send_telegram_alert(f"🚨 Bot error: {str(e)}")
            time.sleep(60)

# === Daily PnL Summary ===
# (functionality replaced with the new summarize_daily_pnl in log_portfolio_snapshot)

def summarize_weekly_pnl():
    try:
        df = pd.read_csv(TRADE_LOG_PATH)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Week'] = df['Timestamp'].dt.to_period('W').astype(str)

        df['SignedAmount'] = df.apply(lambda row: row['Amount'] * row['Price'] if row['Side'] == 'sell' else -row['Amount'] * row['Price'], axis=1)
        summary = df.groupby('Week')['SignedAmount'].sum().reset_index()
        summary.columns = ['Week', 'Net PnL']

        lines = [f"{row['Week']}: ${row['Net PnL']:.2f}" for _, row in summary.iterrows()]
        if lines:
            send_telegram_alert("📆 WEEKLY PNL SUMMARY:\n" + "\n".join(lines))

        # Log to Google Sheet tab: VX-C Weekly PnL
        client = get_gspread_client()
        sheet = None
        if client:
            try:
                sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID"))
            except Exception as sheet_err:
                send_telegram_alert(f"⚠️ Failed to access Google Sheet: {sheet_err}")

        if sheet:
            try:
                try:
                    weekly_tab = sheet.worksheet("VX-C Weekly PnL")
                except Exception:
                    weekly_tab = sheet.add_worksheet(title="VX-C Weekly PnL", rows="1000", cols="3")
                    weekly_tab.append_row(["Week", "Net PnL", "Asset Type"])
                for _, row in summary.iterrows():
                    weekly_tab.append_row([row['Week'], round(row['Net PnL'], 2), "crypto"])
            except Exception as sheet_err:
                send_telegram_alert(f"⚠️ Failed to log VX-C Weekly PnL to sheet: {sheet_err}")

        # Monthly PnL Summary
        df['Month'] = df['Timestamp'].dt.to_period('M').astype(str)
        monthly_summary = df.groupby('Month')['SignedAmount'].sum().reset_index()
        monthly_summary.columns = ['Month', 'Net PnL']
        lines_month = [f"{row['Month']}: ${row['Net PnL']:.2f}" for _, row in monthly_summary.iterrows()]
        if lines_month:
            send_telegram_alert("📅 MONTHLY PNL SUMMARY:\n" + "\n".join(lines_month))

        # All-Time PnL Summary
        total_pnl = df['SignedAmount'].sum()
        send_telegram_alert(f"🧮 ALL-TIME PNL SUMMARY:\nNet PnL: ${total_pnl:.2f}")

        # Log Monthly and All-Time to Google Sheet tab: VX-C Monthly PnL
        try:
            monthly_tab_name = "VX-C Monthly PnL"
            if sheet:
                try:
                    monthly_tab = sheet.worksheet(monthly_tab_name)
                except Exception:
                    monthly_tab = sheet.add_worksheet(title=monthly_tab_name, rows="1000", cols="3")
                    monthly_tab.append_row(["Month", "Net PnL", "Asset Type"])
                for _, row in monthly_summary.iterrows():
                    monthly_tab.append_row([row['Month'], round(row['Net PnL'], 2), "crypto"])
                monthly_tab.append_row(["All Time", round(total_pnl, 2), "crypto"])
        except Exception as sheet_err:
            send_telegram_alert(f"⚠️ Failed to log VX-C Monthly PnL to sheet: {sheet_err}")
    except Exception as e:
        send_telegram_alert(f"⚠️ Failed to summarize weekly PnL: {str(e)}")

if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("🛑 Bot stopped manually.")
    except Exception as e:
        send_telegram_alert(f"🚨 Uncaught error in main loop: {str(e)}")
