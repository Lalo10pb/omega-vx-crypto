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
    # Enforce column order: ["Timestamp", "Symbol", "Side", "Amount", "Price", "Reason"]
    row = [timestamp, symbol, side, amount, price, reason]

    # Log to local CSV
    with open(TRADE_LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Timestamp", "Symbol", "Side", "Amount", "Price", "Reason"])
        writer.writerow(row)

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
trailing_stop_pct = 4.5  # widened to reduce false exits from micro swings
take_profit_pct = 6.0    # increased target to capture stronger breakouts
hard_stop_pct = 4.0      # allows slightly more downside to avoid noise

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

                # Filter: Require at least 1% spread in last 12 closes
                min_price = min(closes[-12:])
                max_price = max(closes[-12:])
                spread_pct = ((max_price - min_price) / min_price) * 100
                if spread_pct < 1.0:
                    print(f"⛔ {symbol} rejected: spread too low ({spread_pct:.2f}%)")
                    continue

                # Filter: Require 1h USD quote volume >= 50k
                try:
                    ticker = kraken.fetch_ticker(symbol)
                    volume_usd = ticker['quoteVolume']
                    if volume_usd is None or volume_usd < 50000:
                        print(f"⛔ {symbol} rejected: low volume (${volume_usd:.0f})")
                        continue
                except Exception as vol_err:
                    print(f"⚠️ Volume fetch error for {symbol}: {vol_err}")
                    continue

                # Scoring logic
                ema_values = calculate_ema(closes, period=20)
                ema_slope = ema_values[-1] - ema_values[-5] if len(ema_values) > 5 else 0
                rsi_values = calculate_rsi(closes, period=14)
                latest_rsi = rsi_values[-1] if rsi_values else 0
                volatility = np.std(closes[-10:]) / np.mean(closes[-10:]) if len(closes) > 10 else 0

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
            # Dynamically determine trade size based on available USD balance
            try:
                balance = kraken.fetch_balance()
                usd_available = balance['USD']['free']
                trade_amount_usd = round(usd_available * 0.05, 2)
                print(f"💰 Dynamic trade amount: ${trade_amount_usd}")
            except Exception as e:
                send_telegram_alert(f"⚠️ Failed to fetch USD balance: {e}")
                trade_amount_usd = 25  # fallback default

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
            summarize_daily_pnl()
            # Weekly PnL summary every Sunday at 5:00 PM (trigger if minute < 5)
            now_dt = datetime.now()
            if now_dt.weekday() == 6 and now_dt.hour == 17 and now_dt.minute < 5:
                summarize_weekly_pnl()
            print("📌 Current open positions:", open_positions)
            time.sleep(15)  # Reduced sleep for faster log testing
        except Exception as e:
            send_telegram_alert(f"🚨 Bot error: {str(e)}")
            time.sleep(60)

# === Daily PnL Summary ===
def summarize_daily_pnl():
    try:
        df = pd.read_csv(TRADE_LOG_PATH)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date

        grouped = df.groupby(['Date', 'Symbol', 'Side'])[['Amount', 'Price']].agg(list).reset_index()
        summary = {}

        for _, row in grouped.iterrows():
            key = (row['Date'], row['Symbol'])
            if key not in summary:
                summary[key] = {'buy': [], 'sell': []}
            summary[key][row['Side']].append((row['Amount'], row['Price']))

        messages = []
        for (day, symbol), sides in summary.items():
            if sides['buy'] and sides['sell']:
                buy_total = sum(a * p for a, p in sides['buy'])
                sell_total = sum(a * p for a, p in sides['sell'])
                pnl = sell_total - buy_total
                messages.append(f"{day} {symbol}: PnL ${pnl:.2f}")

        if messages:
            send_telegram_alert("📊 DAILY TRADE SUMMARY:\n" + "\n".join(messages))

        # Log to Google Sheet tab: VX-C Daily PnL
        try:
            client = get_gspread_client()
            if client:
                sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID"))
                try:
                    daily_tab = sheet.worksheet("VX-C Daily PnL")
                except:
                    daily_tab = sheet.add_worksheet(title="VX-C Daily PnL", rows="1000", cols="5")
                    daily_tab.append_row(["Date", "Symbol", "PnL", "Asset Type"])
                for (day, symbol), sides in summary.items():
                    if sides['buy'] and sides['sell']:
                        buy_total = sum(a * p for a, p in sides['buy'])
                        sell_total = sum(a * p for a, p in sides['sell'])
                        pnl = sell_total - buy_total
                        daily_tab.append_row([str(day), symbol, round(pnl, 2), "crypto"])
        except Exception as sheet_err:
            send_telegram_alert(f"⚠️ Failed to log VX-C Daily PnL to sheet: {sheet_err}")
    except Exception as e:
        send_telegram_alert(f"⚠️ Failed to summarize PnL: {str(e)}")

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
    except Exception as e:
        send_telegram_alert(f"⚠️ Failed to summarize weekly PnL: {str(e)}")

if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("🛑 Bot stopped manually.")
    except Exception as e:
        send_telegram_alert(f"🚨 Uncaught error in main loop: {str(e)}")