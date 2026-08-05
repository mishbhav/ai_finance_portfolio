from agents.base import Agent
from agents.llm_client import call_llm

EXPLAINER_PROMPT = """You are a financial translator. You will be given a
technical investment decision and simulation results. Your only job is to
restate them in plain, jargon-free English for someone with no finance
background — do not add new claims, predictions, or advice beyond what's
given to you. If a term is unavoidable (e.g. "volatility"), briefly define
it in plain words the first time you use it.

Structure your response as:
1. The bottom-line recommendation, in one sentence
2. Why, in 2-3 plain sentences
3. What could go wrong / the honest uncertainty, in 1-2 sentences
Keep the whole thing under 150 words."""

class ExplainerAgent(Agent):
    name = "explainer_agent"
    description = "Translates the judge's decision and simulation results into plain English"

    def run(self, input_data: dict) -> dict:
        query = input_data["query"]
        decision = input_data["decision"]              # judge's synthesis text
        simulation_summary = input_data["simulation_summary"]  # dict from summarize_simulation()

        # Extract parameters safely, falling back to a default if num_simulations isn't present
        num_simulations = simulation_summary.get("num_simulations", 1000)
        p5 = simulation_summary["p5"]
        p50 = simulation_summary["p50"]
        p95 = simulation_summary["p95"]
        probability_of_loss = simulation_summary["probability_of_loss"]

        # Formulate the descriptive narrative phrase requested by the prompt
        simulation_narrative = (
            f"Across {num_simulations} simulated future scenarios, the median outcome was {p50:.1f}. "
            f"The simulation showed a 90% confidence interval ranging from a pessimistic downside of {p5:.1f} "
            f"to an optimistic upside of {p95:.1f}, with a {probability_of_loss * 100:.0f}% chance of "
            f"ending below today's value."
        )

        # Build a structured user message combining all three mandatory elements
        user_message = (
            f"Original Query:\n{query}\n\n"
            f"Judge's Decision:\n{decision}\n\n"
            f"Simulation Summary:\n{simulation_narrative}"
        )

        plain_summary = call_llm(system_prompt=EXPLAINER_PROMPT, user_message=user_message)

        return {
            "agent": self.name,
            "plain_summary": plain_summary,
        }   