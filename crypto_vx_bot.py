import base64
import hashlib
import hmac
import uuid
import json
import ccxt
import os
import time
from collections import defaultdict
from dotenv import load_dotenv

 # Load .env file from project root
env_path = os.path.join(os.path.dirname(__file__), ".env")
print("📦 Attempting to load .env file...")
load_dotenv(dotenv_path=env_path)

print("🔑 Loaded API KEY:", os.getenv("KRAKEN_API_KEY", "[MISSING]"))
print("📁 Looking for .env at:", env_path)
if not os.getenv("KRAKEN_API_KEY"):
    print("❌ ERROR: .env file not loaded correctly or KRAKEN_API_KEY is missing.")
    exit(1)
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
        client = get_gspread_client()
        if client:
            sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID"))
            trade_tab = os.getenv("TRADE_SHEET_NAME", "Crypto Trade Log")
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

def execute_trade(symbol, side, amount, price=None):
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
    if now - last_trade_time[symbol] < TRADE_COOLDOWN_SECONDS:
        wait_min = int((TRADE_COOLDOWN_SECONDS - (now - last_trade_time[symbol])) / 60)
        reason = f"⏳ GLOBAL COOLDOWN: {symbol} trade blocked ({wait_min} min left)."
        print(reason)
        send_telegram_alert(reason)
        return

    try:
        print(f"🛒 Placing {side.upper()} order for {symbol} on {exchange_name.upper()}...")
        if side == "buy":
            price = exchange.fetch_ticker(symbol)['last']
            order = exchange.create_market_buy_order(symbol, amount)
        elif side == "sell":
            order = exchange.create_market_sell_order(symbol, amount)
        else:
            send_telegram_alert(f"❌ Invalid trade side: {side}")
            return

        last_buy_time[symbol] = now if side == "buy" else last_buy_time[symbol]
        last_trade_time[symbol] = now
        if side == "buy":
            open_positions.add(symbol)
            open_positions_data[symbol] = {
                "entry_price": order['average'] or order.get('price'),
                "amount": amount
            }
        elif side == "sell":
            open_positions.discard(symbol)
            open_positions_data.pop(symbol, None)

        price = order['average'] or order.get('price')
        log_trade(symbol, side, amount, price, "Live trade executed")
        print(f"✅ {side.upper()} order executed for {symbol} at ${price:.2f}")
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
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            entry = open_positions_data.get(symbol, {})
            if not entry:
                continue
            entry_price = entry['entry_price']
            amount = entry['amount']
            change_pct = ((current_price - entry_price) / entry_price) * 100

            if change_pct >= take_profit_pct:
                send_telegram_alert(f"🎯 TAKE PROFIT HIT for {symbol} (+{change_pct:.2f}%)")
                execute_trade(symbol, "sell", amount, current_price)
            elif change_pct <= -hard_stop_pct:
                send_telegram_alert(f"🛑 HARD STOP triggered for {symbol} ({change_pct:.2f}%)")
                execute_trade(symbol, "sell", amount, current_price)
            elif change_pct <= -trailing_stop_pct:
                send_telegram_alert(f"📉 TRAILING STOP triggered for {symbol} ({change_pct:.2f}%)")
                execute_trade(symbol, "sell", amount, current_price)
            else:
                print(f"⏳ {symbol} holding: {change_pct:.2f}%")
        except Exception as e:
            send_telegram_alert(f"⚠️ monitor_positions error for {symbol}: {str(e)}")

# === Improved Coin Scanner ===
def scan_top_cryptos(limit=5):
    try:
        print(f"🔍 Scanning {exchange_name.upper()} markets (BEST STRATEGY)...")
        markets = exchange.load_markets()
        scores = []

        # Handle quote currency for Kraken vs others
        if isinstance(exchange, ccxt.kraken):
            quote_currency = "/USD"
        else:
            quote_currency = "/USDT"

        for symbol in markets:
            if not symbol.endswith(quote_currency):
                continue
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
                if len(ohlcv) < 50:
                    continue

                closes = [c[4] for c in ohlcv]
                volumes = [c[5] for c in ohlcv]

                ema9 = calculate_ema(closes, period=9)[-1]
                ema21 = calculate_ema(closes, period=21)[-1]
                ema50 = calculate_ema(closes, period=50)[-1]
                last_price = closes[-1]
                rsi = calculate_rsi(closes)[-1]
                avg_vol = np.mean(volumes[-20:])
                curr_vol = volumes[-1]

                score = 0
                if ema9 > ema21 > ema50:
                    score += 2
                if 40 < rsi < 70:
                    score += 1
                if curr_vol > 2 * avg_vol:
                    score += 1
                if abs(closes[-1] - closes[-5]) / closes[-5] > 0.01:
                    score += 1

                if score >= 3:
                    scores.append((symbol, score, rsi, last_price))
            except Exception:
                continue

        scores.sort(key=lambda x: (-x[1], -x[2]))
        top = [s[0] for s in scores[:limit]]
        print("✅ BEST STRATEGY PICKS:", top)
        return top
    except Exception as e:
        send_telegram_alert(f"❌ Best-strategy scan error: {e}")
        return []

# === Main Bot Loop ===
def run_bot():
    print("🔁 Starting OMEGA-VX-CRYPTO bot loop...")
    send_telegram_alert("🚀 OMEGA-VX-CRYPTO bot started loop")
    while True:
        try:
            # Live trading mode with fixed $10 per trade
            try:
                trade_amount_usd = 10.00  # Live trading with $10 per trade
                print(f"💰 Live trading mode active. Fixed trade amount: ${trade_amount_usd}")
            except Exception as e:
                send_telegram_alert(f"⚠️ Failed to set live trade amount: {e}")
                trade_amount_usd = 10.00

            pairs = scan_top_cryptos()
            send_telegram_alert(f"🧠 Scanned top cryptos: {pairs}")
            for symbol in pairs:
                print(f"📈 Evaluating {symbol}...")
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
                    closes = [candle[4] for candle in ohlcv]
                    price = closes[-1]
                    amount = round(trade_amount_usd / price, 6)
                    print(f"✅ {symbol} passed filters. Forcing test trade.")
                    execute_trade(symbol, "buy", amount)
                except Exception as e:
                    send_telegram_alert(f"⚠️ Error evaluating {symbol}: {str(e)}")

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
        try:
            client = get_gspread_client()
            if client:
                sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID"))
                try:
                    weekly_tab = sheet.worksheet("VX-C Weekly PnL")
                except:
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
            if client:
                try:
                    monthly_tab = sheet.worksheet(monthly_tab_name)
                except:
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