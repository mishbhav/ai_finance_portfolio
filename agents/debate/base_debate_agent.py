from agents.base import Agent
from agents.llm_client import call_llm
from retrieval.retriever import search_with_metadata

MAX_ITERATIONS = 3


class DebateAgent(Agent):
    stance_prompt: str  # overridden per subclass

    def run(self, input_data: dict) -> dict:
        query = input_data["query"]
        revision_feedback = input_data.get("revision_feedback")

        results = search_with_metadata(query, k=3)
        evidence = [r["text"] for r in results]
        evidence_metadata = [r["metadata"] for r in results]

        evidence_block = "\n\n".join(f"- {chunk}" for chunk in evidence)
        user_message = f"Query: {query}\n\nRetrieved evidence:\n{evidence_block}"
        if revision_feedback:
            user_message += (
                f"\n\nA reviewer flagged your previous argument as weakly supported: "
                f"{revision_feedback}\nRevise your argument to rely more strictly on the evidence given."
            )

        argument = call_llm(system_prompt=self.stance_prompt, user_message=user_message)

        return {
            "stance": self.name,
            "argument": argument,
            "evidence": evidence,
            "evidence_metadata": evidence_metadata,
        }