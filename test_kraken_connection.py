import ccxt
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Get keys from env
api_key = os.getenv("KRAKEN_API_KEY")
api_secret = os.getenv("KRAKEN_API_SECRET")

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
