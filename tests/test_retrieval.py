import unittest
import sys
from pathlib import Path
# Calculates the root workspace path relative to this test file and adds it to the search path
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retrieval.retriever import search, get_retriever

class TestRetriever(unittest.TestCase):

    def test_search_returns_results(self):
        """A relevant query should return a non-empty list of chunks."""
        results = search("diversification during a market crash", k=3)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

    def test_search_results_are_strings(self):
        """Each result should be raw text content, not a Document object."""
        results = search("market volatility", k=2)
        for chunk in results:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk), 0)

    def test_k_parameter_limits_results(self):
        """Requesting k=1 should return exactly one chunk (assuming corpus has enough content)."""
        results = search("investing", k=1)
        self.assertEqual(len(results), 1)

    def test_relevance_sanity_check(self):
        """A query about risk should surface content from a risk-related
        source file, not something unrelated. This is a coarse check —
        adjust the keyword to something you know appears in your
        risk_management.txt corpus file."""
        results = search("hedging against downside risk", k=3)
        combined = " ".join(results).lower()
        self.assertTrue(
            any(keyword in combined for keyword in ["risk", "hedge", "volatility", "downside"]),
            "Expected risk-related terms in top results — check chunk_size or corpus content"
        )

if __name__ == "__main__":
    unittest.main()