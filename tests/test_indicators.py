import os
import unittest

# Avoid initializing exchanges and external services when running unit tests.
os.environ["OMEGA_SKIP_INIT"] = "true"

from crypto_vx_bot import calculate_rsi, compute_macd, compute_atr  # noqa: E402


class IndicatorMathTests(unittest.TestCase):
    def test_rsi_reference_values(self):
        prices = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
            45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
            46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
            43.42, 42.66, 43.13,
        ]
        rsi_values = calculate_rsi(prices, period=14)
        self.assertGreater(len(rsi_values), 0)
        self.assertAlmostEqual(rsi_values[0], 70.46413502109705, places=5)
        self.assertAlmostEqual(rsi_values[-1], 37.788771982057824, places=5)

    def test_macd_reference_slice(self):
        prices = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
            45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
            46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
            43.42, 42.66, 43.13, 43.61, 44.00, 44.50, 45.00, 45.50, 46.00, 46.50,
        ]
        macd_line, signal_line, histogram = compute_macd(prices, fast=12, slow=26, signal=9)
        self.assertEqual(len(macd_line), len(signal_line))
        self.assertEqual(len(macd_line), len(histogram))
        self.assertAlmostEqual(macd_line[-1], 0.06291265300856708, places=5)
        self.assertAlmostEqual(signal_line[-1], -0.18813614575324075, places=5)
        self.assertAlmostEqual(histogram[-1], 0.25104879876180786, places=5)

    def test_atr_reference_value(self):
        highs = [1, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 6, 6, 7, 8]
        lows = [0.5, 1, 2, 2, 2.5, 3, 3.5, 4, 3.5, 3.8, 4.2, 4.8, 5, 5.5, 6]
        closes = [0.8, 1.5, 2.5, 2.8, 3.0, 3.8, 4.5, 4.2, 3.9, 4.5, 4.8, 5.2, 5.6, 6.5, 7]

        atr_value = compute_atr(highs, lows, closes, period=14)
        self.assertIsNotNone(atr_value)
        self.assertAlmostEqual(atr_value, 1.2045741627916369, places=5)


if __name__ == "__main__":
    unittest.main()
