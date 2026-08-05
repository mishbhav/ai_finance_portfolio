from agents.debate.bull import BullAgent
from agents.debate.bear import BearAgent
from agents.debate.risk_parity import RiskParityAgent
from agents.judge import JudgeAgent

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

    # TODO: for each weak argument, call its agent's .run() again with
    # revision_feedback=that argument's justification, and replace the
    # corresponding entry in `arguments` with the revised version
    for weak_arg in result["weak_arguments"]:
        agent = agents[weak_arg["stance"]]
        revised_argument = agent.run({
            "query": QUERY,
            "revision_feedback": weak_arg["justification"]
        })
        # Replace the old argument with the revised one
        for i, arg in enumerate(arguments):
            if arg["stance"] == weak_arg["stance"]:
                arguments[i] = revised_argument
                break

print("\nFINAL DECISION:\n", result["decision"])