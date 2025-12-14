import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import bot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock environment variables before importing
with patch.dict(os.environ, {
    "BOT_LOG_LEVEL": "debug", 
    "BotConfig": "MockConfig"
}):
    # Import the module to test
    import crypto_vx_bot
    from crypto_vx_bot import evaluate_market_regime, RegimeState

class TestRegimeLogic(unittest.TestCase):

    def setUp(self):
        # Reset global state before each test
        crypto_vx_bot.regime_last_check = 0.0
        crypto_vx_bot.regime_last_result = RegimeState(50.0, "INIT", "Init", True, 0.5)

    @patch('crypto_vx_bot.fetch_ohlcv_cached')
    def test_strong_bull_market(self, mock_fetch):
        # Create a series of increasing prices
        # 200 candles, linear increase from 100 to 300
        mock_ohlcv = []
        for i in range(200):
            price = 100 + i
            # [timestamp, open, high, low, close, volume]
            mock_ohlcv.append([0, price, price+1, price-1, price, 1000])
        
        mock_fetch.return_value = mock_ohlcv
        
        state = evaluate_market_regime(force=True)
        
        print(f"\nBull Test Score: {state.score}, Status: {state.status}")
        
        # Expect High Score
        # EMA Fast > Slow (+30)
        # Price > EMA200 (+20)
        # ROC Positive (+20 cap)
        # RSI ~ 100 (Overbought) -> maybe penalty? Logic says >80 is just "Overbought" reason, no explicit penalty in code unless i missed it.
        # Wait, code: if 45<=rsi<=75: +10. Else if rsi<30: +5. If >80: just reason.
        # ADX: Trend is strong (+20).
        # Total expected: 30 + 20 + 20 + 0 + 20 = 90.
        
        self.assertEqual(state.status, "BULLISH")
        self.assertGreaterEqual(state.score, 75)
        self.assertTrue(state.can_trade)
        self.assertEqual(state.risk_scaler, 1.0)

    @patch('crypto_vx_bot.fetch_ohlcv_cached')
    def test_bear_market(self, mock_fetch):
        # Decreasing prices
        mock_ohlcv = []
        for i in range(200):
            price = 300 - i
            mock_ohlcv.append([0, price, price+1, price-1, price, 1000])
            
        mock_fetch.return_value = mock_ohlcv
        
        state = evaluate_market_regime(force=True)
        
        print(f"\nBear Test Score: {state.score}, Status: {state.status}")
        
        # Expect Low Score
        # EMA Fast < Slow (0)
        # Price < EMA200 (0)
        # ROC Negative (-10)
        # ADX Strong (but bear trend) -> -20
        # Score likely 0.
        
        self.assertLess(state.score, 20)
        self.assertFalse(state.can_trade)

    @patch('crypto_vx_bot.fetch_ohlcv_cached')
    def test_neutral_chop(self, mock_fetch):
        # Flat prices with noise
        base_price = 100.0
        mock_ohlcv = []
        import math
        for i in range(200):
            # Sine wave to create some movement but flat trend
            price = base_price + math.sin(i/10) * 2
            mock_ohlcv.append([0, price, price+1, price-1, price, 1000])
            
        mock_fetch.return_value = mock_ohlcv
        
        state = evaluate_market_regime(force=True)
        
        print(f"\nChop Test Score: {state.score}, Status: {state.status}")
        
        # Expect Medium Score
        # EMA might be close
        # ROC close to 0
        # ADX low (no trend)
        # RSI oscillating around 50 (+10)
        
        self.assertGreater(state.score, 20)
        self.assertTrue(state.can_trade)

if __name__ == '__main__':
    unittest.main()
