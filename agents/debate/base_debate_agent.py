from agents.base import Agent
from agents.llm_client import call_llm
from retrieval.retriever import search

class DebateAgent(Agent):
    stance_prompt: str   # overridden per subclass

    def run(self, input_data: dict) -> dict:
        query = input_data["query"]

        # 1. ACT (single call, no loop yet): retrieve evidence for the raw query
        evidence = search(query, k=3)

        # 2. Build the user message the LLM will see: the query plus
        #    the retrieved evidence, clearly labeled so the model
        #    can distinguish "the question" from "supporting material"
        evidence_block = "\n\n".join(f"- {chunk}" for chunk in evidence)
        user_message = f"Query: {query}\n\nRetrieved evidence:\n{evidence_block}"

        # 3. Generate the argument using this agent's stance
        argument = call_llm(system_prompt=self.stance_prompt, user_message=user_message)

        return {
            "stance": self.name,
            "argument": argument,
            "evidence": evidence,
        }