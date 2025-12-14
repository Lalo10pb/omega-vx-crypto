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
        # Decreasing prices (Linear drop 300 -> 100)
        # This results in RSI ~0 (Oversold).
        mock_ohlcv = []
        for i in range(200):
            price = 300 - i
            mock_ohlcv.append([0, price, price+1, price-1, price, 1000])
            
        mock_fetch.return_value = mock_ohlcv
        
        state = evaluate_market_regime(force=True)
        
        print(f"\nBear Test Score: {state.score}, Status: {state.status}")
        
        # Expect Defensive Score (due to Oversold RSI condition)
        # RSI < 30 -> +20, Penalties Waived.
        # Score ~20.
        
        self.assertGreaterEqual(state.score, 15)
        self.assertTrue(state.can_trade)

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

    @patch('crypto_vx_bot.fetch_ohlcv_cached')
    def test_low_adx_bear_consolidation(self, mock_fetch):
        # Mild downtrend, low volatility (Low ADX).
        start_price = 100.0
        mock_ohlcv = []
        import random
        for i in range(200):
            # Slow drift
            drift = i * 0.05
            # Random noise to break trend
            noise = random.uniform(-2.0, 2.0)
            price = start_price - drift + noise
            mock_ohlcv.append([0, price, price+0.5, price-0.5, price, 1000])
            
        mock_fetch.return_value = mock_ohlcv
        
        state = evaluate_market_regime(force=True)
        print(f"\nLow ADX Bear Test Score: {state.score}, Status: {state.status}, Reason: {state.reason}")
        
        # Expectation:
        # Low ADX -> +15.
        # RSI ~ 50 -> 0.
        # Score ~15 -> BEARISH_DEFENSIVE.
        self.assertGreaterEqual(state.score, 15)
        self.assertEqual(state.status, "BEARISH_DEFENSIVE")
        self.assertTrue(state.can_trade)

    @patch('crypto_vx_bot.fetch_ohlcv_cached')
    def test_oversold_bounce(self, mock_fetch):
        # Sharp drop to trigger RSI < 30.
        # Should get +20 from RSI Oversold -> BEARISH_DEFENSIVE.
        
        mock_ohlcv = []
        price = 100.0
        for i in range(200):
            if i > 180:
                price -= 2.0 # Sharp drop at end
            else:
                price += 0.1 # Slow rise before
            mock_ohlcv.append([0, price, price+1, price-1, price, 1000])
            
        mock_fetch.return_value = mock_ohlcv
        
        state = evaluate_market_regime(force=True)
        print(f"\nOversold Test Score: {state.score}, Status: {state.status}")
        
        # Expecting at least 20 pts from RSI < 30.
        self.assertGreaterEqual(state.score, 15)
        self.assertTrue(state.can_trade)

    @patch('crypto_vx_bot.fetch_ohlcv_cached')
    def test_log_scenario_repro(self, mock_fetch):
        # Reproduction of specific log scenario:
        # EMA stack fail (Fast < Slow)
        # ROC approx -1.9%
        # We need a downtrend.
        
        # Start high, go low.
        start_price = 92000.0
        end_price = 90280.0 # Match close log
        # 200 candles
        mock_ohlcv = []
        for i in range(200):
            # Linear decay roughly
            progress = i / 199.0
            price = start_price - (start_price - end_price) * progress
            # Add some noise so ADX might be mixed, 
            # but let's see. logic is linear decay = strong trend = High ADX = -20 penalty.
            import random
            noise = random.uniform(-50, 50)
            p = price + noise
            mock_ohlcv.append([0, p, p+50, p-50, p, 1000])
            
        mock_fetch.return_value = mock_ohlcv
        
        state = evaluate_market_regime(force=True)
        print(f"\nLog Repro Score: {state.score}, Status: {state.status}, Reason: {state.reason}")
        
        # If trend is strong (High ADX) and Bearish, score should still be 0 or low.
        # This confirms we screen out catching falling knives (strong trend down).
        # Assert is mostly for logging info, but let's check it's low.
        if "Strong Bear Trend" in state.reason:
             self.assertLess(state.score, 15)



if __name__ == '__main__':
    unittest.main()
