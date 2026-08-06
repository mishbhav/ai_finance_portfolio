import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

SERVER_DIR = str(Path(__file__).resolve().parent.parent / "mcp_servers" / "fundamentals_server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from tools import get_valuation_metrics, get_financial_health, get_fundamentals_summary, get_analyst_sentiment


class TestFundamentalsTools(unittest.TestCase):

    @patch("tools.yf.Ticker")
    def test_get_valuation_metrics_returns_expected_keys(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_instance.info = {
            "trailingPE": 45.2, "forwardPE": 38.1,
            "priceToBook": 12.4, "enterpriseToEbitda": 22.0,
        }
        mock_ticker_cls.return_value = mock_instance

        result = get_valuation_metrics("ASIANPAINT.NS")

        self.assertEqual(result["trailing_pe"], 45.2)
        self.assertFalse(result["trailing_pe_distorted"])

    @patch("tools.yf.Ticker")
    def test_get_valuation_metrics_handles_missing_fields(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_instance.info = {}
        mock_ticker_cls.return_value = mock_instance

        result = get_valuation_metrics("SOME_THIN_TICKER.NS")

        self.assertIsNone(result["trailing_pe"])
        self.assertFalse(result["trailing_pe_distorted"])  # can't be distorted with no data

    @patch("tools.yf.Ticker")
    def test_trailing_pe_distortion_flag_fires_correctly(self, mock_ticker_cls):
        """Mirrors the real GUJENERGY case: trailing PE far above forward PE."""
        mock_instance = MagicMock()
        mock_instance.info = {"trailingPE": 838.3, "forwardPE": 9.3}
        mock_ticker_cls.return_value = mock_instance

        result = get_valuation_metrics("GUJENERGY.NS")

        self.assertTrue(result["trailing_pe_distorted"])

    @patch("tools.yf.Ticker")
    def test_debt_to_equity_is_normalized_to_a_ratio(self, mock_ticker_cls):
        """Mirrors the real SBICARD case: yfinance reports 280.2 (a
        percentage), this should come back as 2.802 (an actual ratio)."""
        mock_instance = MagicMock()
        mock_instance.info = {"debtToEquity": 280.206, "currentRatio": None, "returnOnEquity": None}
        mock_ticker_cls.return_value = mock_instance

        result = get_financial_health("SBICARD.NS")

        self.assertAlmostEqual(result["debt_to_equity"], 2.80206, places=4)

    @patch("tools.yf.Ticker")
    def test_get_fundamentals_summary_shape(self, mock_ticker_cls):
        """Confirms the combined tool nests correctly and only fetches once."""
        mock_instance = MagicMock()
        mock_instance.info = {"trailingPE": 20.0, "forwardPE": 18.0}
        mock_ticker_cls.return_value = mock_instance

        result = get_fundamentals_summary("CRISIL.NS")

        self.assertIn("valuation", result)
        self.assertIn("growth", result)
        self.assertIn("financial_health", result)
        self.assertIn("dividends", result)
        mock_ticker_cls.assert_called_once()  # one yf.Ticker() call, not four

    @patch("tools.yf.Ticker")
    def test_get_fundamentals_summary_shape(self, mock_ticker_cls):
        """Confirms the combined tool nests correctly, including the newer
        analyst_sentiment category. Note: this now makes multiple yf.Ticker()
        calls internally (valuation/growth/health/dividends share one _get_info
        call, but analyst_sentiment's .recommendations access requires a
        separate call) — see get_fundamentals_summary's docstring for why
        this tradeoff was accepted."""
        mock_instance = MagicMock()
        mock_instance.info = {"trailingPE": 20.0, "forwardPE": 18.0}
        mock_instance.recommendations = None
        mock_ticker_cls.return_value = mock_instance

        result = get_fundamentals_summary("CRISIL.NS")

        self.assertIn("valuation", result)
        self.assertIn("growth", result)
        self.assertIn("financial_health", result)
        self.assertIn("dividends", result)
        self.assertIn("analyst_sentiment", result)  # NEW category, worth asserting explicitly
        # No longer asserting a single yf.Ticker() call — analyst_sentiment's
        # .recommendations access is a genuinely separate network operation,
        # not something worth forcing into one call at the cost of complexity.

if __name__ == "__main__":
    unittest.main()