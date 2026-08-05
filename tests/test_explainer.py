from pathlib import Path
import sys

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from agents.explainer import ExplainerAgent

agent = ExplainerAgent()
result = agent.run({
    "query": "should I hold tech stocks right now?",
    "decision": "Based on well-grounded risk-parity evidence showing high correlation "
                "between tech holdings, a partial rebalance is recommended, despite a "
                "moderately-grounded bull case citing strong earnings momentum.",
    "simulation_summary": {
        "expected_value": 108.2, "p5": 89.1, "p50": 107.5, "p95": 131.4,
        "probability_of_loss": 0.22,
    },
})
print(result["plain_summary"])