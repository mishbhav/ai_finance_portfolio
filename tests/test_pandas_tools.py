import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal
import sys
from pathlib import Path

# Automatically calculates the path to your server folder relative to this test file
SERVER_DIR = str(Path(__file__).resolve().parent.parent / "mcp_servers" / "pandas_server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

# Core functions live in tools.py inside mcp_servers/pandas_server/
from tools import (
    calculate_returns,
    calculate_volatility,
    calculate_drawdown,
    calculate_correlation
)


class TestPandasTools(unittest.TestCase):

    def setUp(self):
        """Set up standard predictable tracking vectors for quantitative evaluation."""
        self.prices = pd.Series([100.0, 105.0, 94.5, 103.95], name="Price")

        # Explicit wide tracking matrix for asset cross-correlation tests
        # (4 rows so pct_change() still leaves enough data points to compute variance)
        self.multivariate_prices = pd.DataFrame({
            "Asset_A": [100.0, 105.0, 102.9,  105.987],
            "Asset_B": [100.0, 95.0,  96.9,   93.993]
        })

    def test_calculate_returns(self):
        """Verify daily percentage changes handle initialization boundaries correctly."""
        expected_returns = pd.Series([np.nan, 0.05, -0.10, 0.10], name="Price")
        result = calculate_returns(self.prices)
        assert_series_equal(result, expected_returns)

    def test_calculate_volatility(self):
        """Verify sample standard deviation math ignores early missing data."""
        returns = calculate_returns(self.prices)
        result = calculate_volatility(returns)

        expected_vol = float(returns.std())
        self.assertAlmostEqual(result, expected_vol, places=6)
        # numpy.float64 subclasses float in CPython, but don't rely on that implicitly
        self.assertIsInstance(result, (float, np.floating))

    def test_calculate_volatility_single_value(self):
        """Edge case: a single-value return series has no variance to measure."""
        single_return = pd.Series([0.05])
        result = calculate_volatility(single_return)
        # std of a single sample is NaN with default ddof=1 — assert your
        # implementation's actual documented behavior here once decided
        self.assertTrue(pd.isna(result) or result == 0.0)

    def test_calculate_drawdown(self):
        """Verify performance drop tracking accurately benchmarks structural peak degradation."""
        # t0: 100 (Peak=100)   -> DD = 0%
        # t1: 105 (Peak=105)   -> DD = 0%
        # t2: 94.5 (Peak=105)  -> DD = (94.5 - 105) / 105  = -10%
        # t3: 103.95 (Peak=105)-> DD = (103.95 - 105) / 105 = -1%
        expected_drawdown = pd.Series([0.0, 0.0, -0.10, -0.01], name="Price")
        result = calculate_drawdown(self.prices)
        assert_series_equal(result, expected_drawdown, atol=1e-6)

    def test_calculate_correlation(self):
        """Verify multi-asset wide frame processing yields a proper square correlation matrix."""
        expected_corr = pd.DataFrame(
            [[1.0, -1.0], [-1.0, 1.0]],
            columns=["Asset_A", "Asset_B"],
            index=["Asset_A", "Asset_B"]
        )
        result = calculate_correlation(self.multivariate_prices)
        assert_frame_equal(result, expected_corr, atol=1e-2)


if __name__ == "__main__":
    unittest.main()