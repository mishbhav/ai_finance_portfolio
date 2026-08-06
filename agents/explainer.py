from agents.base import Agent
from agents.llm_client import call_llm

EXPLAINER_PROMPT = """You are a financial translator. You will be given a
technical investment decision and simulation results, along with what the
simulation was scoped to (one specific holding, or the whole portfolio).
Your only job is to restate them in plain, jargon-free English for someone
with no finance background — do not add new claims, predictions, or advice
beyond what's given to you. Always state clearly whether the simulation
numbers reflect the single holding being discussed or the entire portfolio
— this distinction matters and should not be glossed over. If a term is
unavoidable (e.g. "volatility"), briefly define it in plain words the
first time you use it.

Structure your response as:
1. The bottom-line recommendation, in one sentence
2. Why, in 2-3 plain sentences
3. What could go wrong / the honest uncertainty, in 1-2 sentences,
   including which scope (single holding vs whole portfolio) the numbers reflect
Keep the whole thing under 150 words."""


class ExplainerAgent(Agent):
    name = "explainer_agent"
    description = "Translates the judge's decision and simulation results into plain English"

    def run(self, input_data: dict) -> dict:
        query = input_data["query"]
        decision = input_data["decision"]
        simulation_summary = input_data["simulation_summary"]
        scope = input_data.get("simulation_scope", {"scope": "portfolio", "ticker": None})

        num_simulations = simulation_summary.get("num_simulations", 1000)
        p5 = simulation_summary["p5"]
        p50 = simulation_summary["p50"]
        p95 = simulation_summary["p95"]
        probability_of_loss = simulation_summary["probability_of_loss"]

        scope_description = (
            f"just the holding {scope['ticker']}" if scope["scope"] == "single_ticker"
            else "the entire portfolio (all holdings blended)"
        )

        simulation_narrative = (
            f"Across {num_simulations} simulated future scenarios of {scope_description}, "
            f"the median outcome was {p50:.1f}. The simulation showed a 90% confidence interval "
            f"ranging from a pessimistic downside of {p5:.1f} to an optimistic upside of {p95:.1f}, "
            f"with a {probability_of_loss * 100:.0f}% chance of ending below today's value."
        )

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