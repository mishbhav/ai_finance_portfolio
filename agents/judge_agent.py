from agents.base import Agent
from judge import score_argument, synthesize_decision

class JudgeAgent(Agent):
    name = "judge_agent"
    description = "Weighs debate arguments by groundedness and synthesizes a decision"

    REVISION_THRESHOLD = 6  # scores below this trigger a revision request

    def run(self, input_data: dict) -> dict:
        query = input_data["query"]
        arguments = input_data["arguments"]  # list of {"stance", "argument", "evidence"}

        scored = []
        for arg in arguments:
            result = score_argument(arg["argument"], arg["evidence"])
            scored.append({**arg, **result})

        weak = [a for a in scored if a["score"] < self.REVISION_THRESHOLD]
        decision = synthesize_decision(query, scored)

        return {
            "scored_arguments": scored,
            "weak_arguments": weak,   # caller decides whether to request revisions
            "decision": decision,
        }