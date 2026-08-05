from agents.llm_client import call_llm
from agents.debate.bull import BullAgent
from agents.debate.bear import BearAgent
from agents.debate.risk_parity import RiskParityAgent
from agents.judge import JudgeAgent
from agents.data_analyst import DataAnalystAgent
from agents.explainer import ExplainerAgent
from simulation.monte_carlo import simulate_price_paths, summarize_simulation
from memory.decision_log import log_decision, read_recent_decisions

import pandas as pd
import json
import re

MAX_REVISION_ROUNDS = 2

PLANNING_PROMPT = """You are a task planner for a portfolio analysis system.
Given a user's query, decide which of these two plans is appropriate:

- "quick_lookup": the user wants a factual number or metric about their
  portfolio (e.g. "what's my volatility", "how much do I have in tech")
  with no decision or recommendation needed.
- "full_debate": the user is asking for a decision or recommendation
  (e.g. "should I rebalance", "is now a good time to sell") that requires
  weighing arguments and running a simulation.

If recent past decisions are provided, use them only as context — they
do not override your judgment on the current query.

Respond with ONLY valid JSON: {"plan": "quick_lookup" or "full_debate", "reasoning": "<one sentence>"}"""

FALLBACK_PLAN = {"plan": "full_debate", "reasoning": "Defaulted to the thorough path after a planning error."}


def _format_history(recent_decisions: list[dict]) -> str:
    if not recent_decisions:
        return ""
    lines = []
    for d in recent_decisions:
        decision_snippet = (d.get("decision") or "")[:100]
        lines.append(f"- \"{d['query']}\" → {decision_snippet}...")
    return "\n\nRecent past decisions (for context, not necessarily relevant):\n" + "\n".join(lines)


def plan_query(query: str, recent_decisions: list[dict] = None) -> dict:
    """Decides which execution path a query needs. Falls back to
    'full_debate' on any parsing failure — a missed quick_lookup just costs
    extra compute, but a missed full_debate could silently skip real
    decision analysis. Better to over-deliver than under-deliver."""
    history_block = _format_history(recent_decisions)
    user_message = query + history_block

    try:
        raw_response = call_llm(system_prompt=PLANNING_PROMPT, user_message=user_message)
        cleaned = raw_response.strip()

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        stripped = match.group(1) if match else cleaned

        parsed = json.loads(stripped)

        if not isinstance(parsed, dict) or "plan" not in parsed:
            raise ValueError("Missing 'plan' key")
        if parsed["plan"] not in ("quick_lookup", "full_debate"):
            raise ValueError(f"Unexpected plan value: {parsed['plan']}")

        return {
            "plan": parsed["plan"],
            "reasoning": str(parsed.get("reasoning", "")),
        }

    except Exception as e:
        print(f"[orchestrator] planning failed, defaulting to full_debate: {e}")
        return FALLBACK_PLAN.copy()


def _get_portfolio_returns(holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Builds a single value-weighted daily return series for the whole
    portfolio, using each holding's most recent price * shares as its
    weight. This is what the Monte Carlo layer simulates forward."""
    held_tickers = holdings["ticker"].tolist()
    held_prices = prices[held_tickers]

    latest_prices = held_prices.iloc[-1]
    shares = holdings.set_index("ticker")["shares"]
    weights = latest_prices * shares
    weights = weights / weights.sum()

    daily_returns = held_prices.pct_change().dropna()
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    return portfolio_returns


class Orchestrator:
    def __init__(self):
        self.debate_agents = {a.name: a for a in [BullAgent(), BearAgent(), RiskParityAgent()]}
        self.judge = JudgeAgent()
        self.data_analyst = DataAnalystAgent()
        self.explainer = ExplainerAgent()

    def run(self, query: str) -> dict:
        recent = read_recent_decisions(limit=5)
        plan = plan_query(query, recent_decisions=recent)
        print(f"[orchestrator] plan: {plan['plan']} ({plan['reasoning']})")

        if plan["plan"] == "quick_lookup":
            return self._run_quick_lookup(query)

        result = self._run_full_debate(query)
        log_decision(query, result)
        return result

    def _run_quick_lookup(self, query: str) -> dict:
        result = self.data_analyst.run({"query": query})
        return {
            "plan": "quick_lookup",
            "result": result,
        }

    def _run_full_debate(self, query: str) -> dict:
        arguments = [agent.run({"query": query}) for agent in self.debate_agents.values()]

        judge_result = None
        for round_num in range(MAX_REVISION_ROUNDS):
            judge_result = self.judge.run({"query": query, "arguments": arguments})

            if not judge_result["weak_arguments"]:
                break

            weak_stances = [a["stance"] for a in judge_result["weak_arguments"]]
            print(f"[orchestrator] round {round_num + 1}: revising {weak_stances}")

            for weak_arg in judge_result["weak_arguments"]:
                stance = weak_arg["stance"]
                agent = self.debate_agents[stance]
                revised = agent.run({
                    "query": query,
                    "revision_feedback": weak_arg["justification"],
                })
                for i, arg in enumerate(arguments):
                    if arg["stance"] == stance:
                        arguments[i] = revised
                        break

        holdings = pd.read_csv("data/sample_portfolio.csv")
        prices = pd.read_csv("data/price_history.csv", index_col=0, parse_dates=True)
        portfolio_returns = _get_portfolio_returns(holdings, prices)

        paths = simulate_price_paths(portfolio_returns, num_simulations=1000, num_days=252, initial_value=100.0)
        simulation_summary = summarize_simulation(paths)

        explainer_result = self.explainer.run({
            "query": query,
            "decision": judge_result["decision"],
            "simulation_summary": simulation_summary,
        })

        return {
            "plan": "full_debate",
            "decision": judge_result["decision"],
            "scored_arguments": judge_result["scored_arguments"],
            "simulation_summary": simulation_summary,
            "plain_summary": explainer_result["plain_summary"],
        }