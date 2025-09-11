import ccxt
import os
import time
from collections import defaultdict
from dotenv import load_dotenv
import numpy as np
import requests
import csv
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
print("🟢 OMEGA-VX-CRYPTO bot started.")
open_positions = set()
open_positions_data = {}
last_buy_time = defaultdict(lambda: 0)
last_trade_time = defaultdict(lambda: 0)
COOLDOWN_SECONDS = 30 * 60  # 30 minutes
TRADE_COOLDOWN_SECONDS = 60 * 60  # 1 hour global cooldown per symbol

 # === CONFIGURATION ===
trade_amount_usd = 5  # USD amount per trade (adjust as needed)

# === Paths ===
TRADE_LOG_PATH = "crypto_trade_log.csv"
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
    rs = up / down if down != 0 else 0
    rsi = [100. - 100. / (1. + rs)]

    for delta in deltas[period:]:
        up_val = max(delta, 0)
        down_val = -min(delta, 0)
        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period
        rs = up / down if down != 0 else 0
        rsi.append(100. - 100. / (1. + rs))
    return rsi

# Load environment variables
load_dotenv()

# === Google Sheets Helper ===
def get_gspread_client():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('google_credentials.json', scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        send_telegram_alert(f"❌ Google Sheets auth error: {str(e)}")
        return None

# Load Kraken API keys
api_key = os.getenv("KRAKEN_API_KEY")
api_secret = os.getenv("KRAKEN_API_SECRET")

# Initialize Kraken client
kraken = ccxt.kraken({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})

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

def log_trade(symbol, side, amount, price, reason):
    timestamp = datetime.now().isoformat()
    row = [timestamp, symbol, side, amount, price, reason]

    # Log to local CSV
    with open(TRADE_LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    # Log to Google Sheet
    try:
        client = get_gspread_client()
        if client:
            sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID")).worksheet(os.getenv("TRADE_SHEET_NAME"))
            sheet.append_row(row)
    except Exception as e:
        send_telegram_alert(f"⚠️ Failed to log trade to sheet: {str(e)}")

    send_telegram_alert(f"📒 LOGGED TRADE: {side.upper()} {symbol} | Amount: {amount} @ ${price:.2f} ({reason})")

def log_portfolio_snapshot():
    try:
        balance = kraken.fetch_balance()
        usd = balance['USD']['free']
        total = sum(asset['total'] for asset in balance['total'].values() if isinstance(asset, dict))
        timestamp = datetime.now().isoformat()

        # Log to local CSV
        with open(PORTFOLIO_LOG_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, usd, total])

        # Log to Google Sheet
        try:
            client = get_gspread_client()
            if client:
                sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID")).worksheet(os.getenv("PORTFOLIO_SHEET_NAME"))
                sheet.append_row([timestamp, usd, total])
        except Exception as sheet_err:
            send_telegram_alert(f"⚠️ Failed to log portfolio to sheet: {str(sheet_err)}")

        print(f"💾 Snapshot: USD=${usd:.2f}, Total=${total:.2f}")
    except Exception as e:
        send_telegram_alert(f"⚠️ Failed to log portfolio snapshot: {str(e)}")

def execute_trade(symbol, side, amount, price=None):
    # Prevent duplicate buys or trades within cooldown window
    now = time.time()
    # Apply global cooldown for both buy/sell
    if now - last_trade_time[symbol] < TRADE_COOLDOWN_SECONDS:
        wait_min = int((TRADE_COOLDOWN_SECONDS - (now - last_trade_time[symbol])) / 60)
        reason = f"⏳ GLOBAL COOLDOWN: {symbol} trade blocked ({wait_min} min left)."
        print(reason)
        send_telegram_alert(reason)
        return
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
    try:
        order = kraken.create_market_order(symbol, side, float(amount))
        send_telegram_alert(f"✅ {side.upper()} order executed for {symbol} | Amount: {amount}")
        last_trade_time[symbol] = now
        log_trade(symbol, side, amount, order['price'] if 'price' in order else 'MKT', "manual trade")
        if side == "buy":
            open_positions.add(symbol)
            last_buy_time[symbol] = time.time()
        return order
    except Exception as e:
        send_telegram_alert(f"❌ Failed to execute {side} order for {symbol}: {str(e)}")

# === SELL LOGIC CONFIG ===
trailing_stop_pct = 2.5  # percent drawdown from peak after entry
take_profit_pct = 4.0    # hard TP
hard_stop_pct = 3.0      # hard SL

# === Monitor & Auto-Close Open Positions ===
def monitor_positions():
    print("🔍 Monitoring live positions...")
    try:
        balance = kraken.fetch_balance()
        tickers = kraken.fetch_tickers()

        for symbol, market_data in tickers.items():
            if "/USD" not in symbol:
                continue

            market = kraken.market(symbol)
            base = market['base']
            quote = market['quote']

            if quote != 'USD' or base not in balance or balance[base]['total'] == 0:
                continue

            amount_held = balance[base]['total']
            current_price = market_data['last']

            # Fetch latest buy trade as pseudo-entry price
            entry_price = None
            try:
                trades = kraken.fetch_my_trades(symbol)
                buys = [t for t in trades if t['side'] == 'buy']
                if buys:
                    entry_price = buys[-1]['price']
            except Exception as e:
                send_telegram_alert(f"⚠️ Trade history error for {symbol}: {e}")
                continue

            if not entry_price:
                continue

            change_pct = ((current_price - entry_price) / entry_price) * 100
            print(f"📊 {symbol}: entry=${entry_price:.2f}, now=${current_price:.2f}, Δ={change_pct:.2f}%")

            # TAKE PROFIT
            if change_pct >= take_profit_pct:
                kraken.create_market_order(symbol, "sell", amount_held)
                log_trade(symbol, "sell", amount_held, current_price, "take profit")
                send_telegram_alert(f"🎯 TAKE PROFIT: Sold {symbol} at ${current_price:.2f} (+{change_pct:.2f}%)")
                open_positions.discard(symbol)
                open_positions_data.pop(symbol, None)
                continue

            # HARD STOP
            if change_pct <= -hard_stop_pct:
                kraken.create_market_order(symbol, "sell", amount_held)
                log_trade(symbol, "sell", amount_held, current_price, "hard stop")
                send_telegram_alert(f"🛑 HARD STOP: Sold {symbol} at ${current_price:.2f} ({change_pct:.2f}%)")
                open_positions.discard(symbol)
                open_positions_data.pop(symbol, None)
                continue

            # TRAILING STOP
            if symbol not in open_positions_data:
                open_positions_data[symbol] = {
                    'entry': entry_price,
                    'peak': current_price
                }
            else:
                if current_price > open_positions_data[symbol]['peak']:
                    open_positions_data[symbol]['peak'] = current_price

                peak = open_positions_data[symbol]['peak']
                drop_pct = ((peak - current_price) / peak) * 100

                if drop_pct >= trailing_stop_pct:
                    kraken.create_market_order(symbol, "sell", amount_held)
                    log_trade(symbol, "sell", amount_held, current_price, "trailing stop")
                    send_telegram_alert(f"🔻 TRAILING STOP: Sold {symbol} at ${current_price:.2f} (-{drop_pct:.2f}% from peak)")
                    open_positions.discard(symbol)
                    open_positions_data.pop(symbol, None)

    except Exception as e:
        send_telegram_alert(f"❌ monitor_positions() error: {e}")

# === Improved Coin Scanner ===
def scan_top_cryptos(limit=5):
    print("🔎 Scanning top crypto pairs by volume and filters...")
    try:
        markets = kraken.load_markets()
        ranked = []

        for symbol, data in markets.items():
            if not data.get('active', False) or "/USD" not in symbol or not data.get('spot', False):
                continue

            try:
                ohlcv = kraken.fetch_ohlcv(symbol, timeframe='1h', limit=50)
                closes = [candle[4] for candle in ohlcv]

                ema_values = calculate_ema(closes, period=20)
                ema_slope = ema_values[-1] - ema_values[-5] if len(ema_values) > 5 else 0
                rsi_values = calculate_rsi(closes, period=14)
                latest_rsi = rsi_values[-1] if rsi_values else 0
                volatility = np.std(closes[-10:]) / np.mean(closes[-10:]) if len(closes) > 10 else 0

                # Scoring logic
                score = 0
                if ema_slope > 0:
                    score += 1
                if 55 <= latest_rsi <= 75:
                    score += 1
                if 0.01 <= volatility <= 0.05:
                    score += 1

                ranked.append({
                    "symbol": symbol,
                    "score": score
                })
                print(f"🔬 {symbol} — Score: {score} | EMA Slope: {ema_slope:.4f} | RSI: {latest_rsi:.2f} | Volatility: {volatility:.4f}")
            except Exception as inner_e:
                print(f"⚠️ Error scoring {symbol}: {inner_e}")
                continue

        ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
        top_symbols = [item["symbol"] for item in ranked[:limit]]
        print(f"✅ AI-ranked top {limit} symbols: {top_symbols}")
        return top_symbols

    except Exception as e:
        send_telegram_alert(f"❌ Scanner error: {str(e)}")
        return []

# === Main Bot Loop ===
def run_bot():
    print("🔁 Starting OMEGA-VX-CRYPTO bot loop...")
    send_telegram_alert("🚀 OMEGA-VX-CRYPTO bot started loop")
    while True:
        try:
            pairs = scan_top_cryptos()
            send_telegram_alert(f"🧠 Scanned top cryptos: {pairs}")
            for symbol in pairs:
                print(f"📈 Evaluating {symbol}...")

                try:
                    ohlcv = kraken.fetch_ohlcv(symbol, timeframe='1h', limit=100)
                    closes = [candle[4] for candle in ohlcv]

                    ema_values = calculate_ema(closes, period=20)
                    if closes[-1] < ema_values[-1]:
                        print(f"⛔ {symbol} rejected: price below EMA.")
                        send_telegram_alert(f"⛔ {symbol} rejected: price below EMA.")
                        continue

                    rsi_values = calculate_rsi(closes, period=14)
                    if rsi_values[-1] < 50:
                        print(f"⛔ {symbol} rejected: RSI below 50.")
                        send_telegram_alert(f"⛔ {symbol} rejected: RSI below 50.")
                        continue

                    print(f"✅ {symbol} passed filters. Ready for trade logic.")

                    try:
                        market = kraken.market(symbol)
                        price = closes[-1]
                        precision = int(market['precision'].get('amount', 6) or 6)
                        amount = round(trade_amount_usd / price, precision)

                        print(f"🛒 Executing LIVE BUY for {symbol} at ${price:.2f}, amount={amount}")
                        execute_trade(symbol, "buy", amount)

                    except Exception as trade_error:
                        send_telegram_alert(f"❌ Trade failed for {symbol}: {str(trade_error)}")

                except Exception as e:
                    send_telegram_alert(f"⚠️ Error evaluating {symbol}: {str(e)}")

            monitor_positions()
            log_portfolio_snapshot()
            print("📌 Current open positions:", open_positions)
            time.sleep(15)  # Reduced sleep for faster log testing
        except Exception as e:
            send_telegram_alert(f"🚨 Bot error: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("🛑 Bot stopped manually.")
    except Exception as e:
        send_telegram_alert(f"🚨 Uncaught error in main loop: {str(e)}")