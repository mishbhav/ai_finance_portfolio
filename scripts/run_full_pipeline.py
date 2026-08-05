"""
End-to-end CLI runner for the Portfolio War Room pipeline.

Usage:
    python scripts/run_full_pipeline.py "should I rebalance out of tech this week?"
    python scripts/run_full_pipeline.py "what's my portfolio's volatility?"

Run from the project root so `agents`, `retrieval`, `simulation`, and
`memory` resolve as packages.
"""
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.orchestrator import Orchestrator


def main():
    if len(sys.argv) < 2:
        query = "should I rebalance out of tech this week?"
        print(f"No query given, using default: {query!r}\n")
    else:
        query = " ".join(sys.argv[1:])

    orchestrator = Orchestrator()
    result = orchestrator.run(query)

    print("\n" + "=" * 60)
    if result["plan"] == "quick_lookup":
        print("PLAN: quick_lookup")
        print(result["result"])
    else:
        print("PLAN: full_debate")
        print("\n--- Scored arguments ---")
        for arg in result["scored_arguments"]:
            print(f"\n[{arg['stance']}] score={arg['score']}/10")
            print(f"  {arg['argument']}")
            print(f"  judge note: {arg['justification']}")

        print("\n--- Judge decision ---")
        print(result["decision"])

        print("\n--- Simulation summary ---")
        sim = result["simulation_summary"]
        print(f"  expected: {sim['expected_value']:.1f}")
        print(f"  p5 / p50 / p95: {sim['p5']:.1f} / {sim['p50']:.1f} / {sim['p95']:.1f}")
        print(f"  probability of loss: {sim['probability_of_loss'] * 100:.0f}%")

        print("\n--- Plain-English summary ---")
        print(result["plain_summary"])
    print("=" * 60)


if __name__ == "__main__":
    main()