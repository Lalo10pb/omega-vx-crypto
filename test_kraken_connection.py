import os
from pathlib import Path

import ccxt
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("KRAKEN_API_KEY")
api_secret = os.getenv("KRAKEN_API_SECRET")

if not api_key or not api_secret:
    print("❌ Missing KRAKEN_API_KEY or KRAKEN_API_SECRET in environment.")
    print(f"ℹ️ Checked .env at: {ENV_PATH}")
    raise SystemExit(1)

# Setup Kraken client
kraken = ccxt.kraken({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})

# Test: fetch balance
try:
    balance = kraken.fetch_balance()
    print("✅ Connection successful!")
    print("Available balance:")
    for currency, info in balance['total'].items():
        if info > 0:
            print(f"{currency}: {info}")
except Exception as e:
    print("❌ Connection failed:", e)
