from agents.llm_client import call_llm
from agents.debate.bull import BullAgent
from agents.debate.bear import BearAgent
from agents.debate.risk_parity import RiskParityAgent
from agents.judge import JudgeAgent
from agents.data_analyst import DataAnalystAgent
from agents.explainer import ExplainerAgent
from simulation.monte_carlo import simulate_price_paths, summarize_simulation
from memory.decision_log import log_decision, read_recent_decisions
from scripts.refresh_corpus import refresh

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

SCOPE_PROMPT_TEMPLATE = """You are a query-scope classifier for a portfolio
analysis system. The user holds these tickers: {tickers}

Given a query, decide whether it's asking about ONE specific holding, or
about the portfolio as a whole (including questions that mention a sector
or theme spanning multiple holdings, e.g. "should I rebalance out of tech").

Respond with ONLY valid JSON:
{{"scope": "single_ticker", "ticker": "<exact ticker from the list above>"}}
or
{{"scope": "portfolio", "ticker": null}}"""

FALLBACK_SCOPE = {"scope": "portfolio", "ticker": None}


def _format_history(recent_decisions: list[dict]) -> str:
    if not recent_decisions:
        return ""
    lines = []
    for d in recent_decisions:
        decision_snippet = (d.get("decision") or "")[:100]
        lines.append(f"- \"{d['query']}\" → {decision_snippet}...")
    return "\n\nRecent past decisions (for context, not necessarily relevant):\n" + "\n".join(lines)


def plan_query(query: str, recent_decisions: list[dict] = None) -> dict:
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


def identify_query_scope(query: str, held_tickers: list[str]) -> dict:
    prompt = SCOPE_PROMPT_TEMPLATE.format(tickers=", ".join(held_tickers))

    try:
        raw_response = call_llm(system_prompt=prompt, user_message=query)
        cleaned = raw_response.strip()

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        stripped = match.group(1) if match else cleaned

        parsed = json.loads(stripped)

        if not isinstance(parsed, dict) or "scope" not in parsed:
            raise ValueError("Missing 'scope' key")
        if parsed["scope"] not in ("single_ticker", "portfolio"):
            raise ValueError(f"Unexpected scope value: {parsed['scope']}")

        ticker = parsed.get("ticker")
        if parsed["scope"] == "single_ticker":
            if ticker not in held_tickers:
                raise ValueError(f"LLM named ticker '{ticker}' not in held tickers")
        else:
            ticker = None

        return {"scope": parsed["scope"], "ticker": ticker}

    except Exception as e:
        print(f"[orchestrator] scope detection failed, defaulting to portfolio: {e}")
        return FALLBACK_SCOPE.copy()


def _get_portfolio_returns(holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    held_tickers = holdings["ticker"].tolist()
    held_prices = prices[held_tickers]

    latest_prices = held_prices.iloc[-1]
    shares = holdings.set_index("ticker")["shares"]
    weights = latest_prices * shares
    weights = weights / weights.sum()

    daily_returns = held_prices.pct_change().dropna()
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    return portfolio_returns


def _get_ticker_returns(ticker: str, prices: pd.DataFrame) -> pd.Series:
    return prices[ticker].pct_change().dropna()


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
        holdings = pd.read_csv("data/sample_portfolio.csv")
        held_tickers = holdings["ticker"].tolist()

        try:
            added = refresh(held_tickers)
            print(f"[orchestrator] news refresh: {added} chunks added")
        except Exception as e:
            print(f"[orchestrator] news refresh failed, continuing with existing corpus: {e}")

        scope = identify_query_scope(query, held_tickers)
        print(f"[orchestrator] simulation scope: {scope['scope']} ({scope.get('ticker') or 'whole portfolio'})")

        arguments = [agent.run({"query": query}) for agent in self.debate_agents.values()]

        judge_result = None
        revision_log = []
        for round_num in range(MAX_REVISION_ROUNDS):
            judge_result = self.judge.run({"query": query, "arguments": arguments})

            if not judge_result["weak_arguments"]:
                break

            weak_stances = [a["stance"] for a in judge_result["weak_arguments"]]
            revision_log.append(f"Round {round_num + 1}: revised {', '.join(weak_stances)}")
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

        prices = pd.read_csv("data/price_history.csv", index_col=0, parse_dates=True)

        if scope["scope"] == "single_ticker":
            portfolio_returns = _get_ticker_returns(scope["ticker"], prices)
        else:
            portfolio_returns = _get_portfolio_returns(holdings, prices)

        paths = simulate_price_paths(portfolio_returns, num_simulations=1000, num_days=252, initial_value=100.0)
        simulation_summary = summarize_simulation(paths)

        explainer_result = self.explainer.run({
            "query": query,
            "decision": judge_result["decision"],
            "simulation_summary": simulation_summary,
            "simulation_scope": scope,
        })

        return {
            "plan": "full_debate",
            "decision": judge_result["decision"],
            "scored_arguments": judge_result["scored_arguments"],
            "simulation_summary": simulation_summary,
            "simulation_scope": scope,
            "revision_log": revision_log,
            "plain_summary": explainer_result["plain_summary"],
        }