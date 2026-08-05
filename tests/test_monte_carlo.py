import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 1. Calculate the root workspace path (one level up from 'tests') and add it to sys.path
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from simulation.monte_carlo import simulate_price_paths, summarize_simulation

class TestMonteCarlo(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)  # makes "random" results reproducible for testing
        # Synthetic returns: small, roughly realistic daily moves
        self.returns = pd.Series(np.random.normal(0.0005, 0.01, 500))

    def test_output_shape(self):
        paths = simulate_price_paths(self.returns, num_simulations=100, num_days=50, initial_value=100.0)
        self.assertEqual(paths.shape, (100, 50))

    def test_starts_near_initial_value(self):
        """Day-1 simulated values should be close to initial_value — not
        wildly off — since only one day's return has been applied."""
        paths = simulate_price_paths(self.returns, num_simulations=100, num_days=50, initial_value=100.0)
        self.assertTrue(np.all(paths[:, 0] > 80) and np.all(paths[:, 0] < 120))

    def test_summary_keys_present(self):
        paths = simulate_price_paths(self.returns, num_simulations=100, num_days=50, initial_value=100.0)
        summary = summarize_simulation(paths)
        for key in ["expected_value", "p5", "p50", "p95", "probability_of_loss"]:
            self.assertIn(key, summary)

    def test_percentile_ordering(self):
        """A basic sanity invariant: p5 <= p50 <= p95 must always hold,
        regardless of the random seed."""
        paths = simulate_price_paths(self.returns, num_simulations=500, num_days=100, initial_value=100.0)
        summary = summarize_simulation(paths)
        self.assertLessEqual(summary["p5"], summary["p50"])
        self.assertLessEqual(summary["p50"], summary["p95"])

if __name__ == "__main__":
    unittest.main()