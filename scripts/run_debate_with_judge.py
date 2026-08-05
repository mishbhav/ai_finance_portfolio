import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.debate.bull import BullAgent
from agents.debate.bear import BearAgent
from agents.debate.risk_parity import RiskParityAgent
from agents.judge_agent import JudgeAgent

MAX_REVISION_ROUNDS = 2
QUERY = "should I hold tech stocks right now?"

agents = {a.name: a for a in [BullAgent(), BearAgent(), RiskParityAgent()]}
arguments = [agent.run({"query": QUERY}) for agent in agents.values()]

judge = JudgeAgent()
for round_num in range(MAX_REVISION_ROUNDS):
    result = judge.run({"query": QUERY, "arguments": arguments})
    if not result["weak_arguments"]:
        break
    print(f"Round {round_num + 1}: revising {[a['stance'] for a in result['weak_arguments']]}")

    for weak_arg in result["weak_arguments"]:
        agent = agents[weak_arg["stance"]]
        revised_argument = agent.run({
            "query": QUERY,
            "revision_feedback": weak_arg["justification"]
        })
        for i, arg in enumerate(arguments):
            if arg["stance"] == weak_arg["stance"]:
                arguments[i] = revised_argument
                break

print("\nFINAL DECISION:\n", result["decision"])