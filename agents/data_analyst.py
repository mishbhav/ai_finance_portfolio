from agents.base import Agent
import pandas as pd
import json
import re
import sys
import importlib.util
from pathlib import Path

# Pandas tools: plain sys.path insert + bare import is safe here since
# nothing else in this file already claims the module name "tools".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp_servers" / "pandas_server"))
from tools import calculate_returns, calculate_volatility, calculate_drawdown, calculate_correlation

# Fundamentals tools: loaded via importlib under a distinct module name
# to avoid colliding with the "tools" import above — same fix as the
# test files needed once two MCP servers both have a module named tools.py.
_fund_spec = importlib.util.spec_from_file_location(
    "fundamentals_tools",
    Path(__file__).resolve().parent.parent / "mcp_servers" / "fundamentals_server" / "tools.py",
)
fundamentals_tools = importlib.util.module_from_spec(_fund_spec)
_fund_spec.loader.exec_module(fundamentals_tools)


TOOL_SELECTION_PROMPT = """You are a quantitative analysis planner. Given a
user's query about their portfolio, decide which of these analyses are
relevant. Only include ones that would actually help answer the query.

Available analyses:
- "volatility": price volatility per holding
- "drawdown": peak-to-trough decline per holding
- "correlation": how holdings move together (only meaningful with 2+ holdings)
- "fundamentals": valuation, growth, financial health, dividends per holding

Respond with ONLY valid JSON: {"analyses": ["volatility", ...], "reasoning": "<one sentence>"}"""

FALLBACK_ANALYSES = {"analyses": ["volatility", "drawdown"], "reasoning": "Defaulted after a planning error."}


class DataAnalystAgent(Agent):
    name = "data_analyst_agent"
    description = "Computes portfolio metrics using pandas and fundamentals data, choosing analyses relevant to the query"

    def run(self, input_data: dict) -> dict:
        from agents.llm_client import call_llm  # local import avoids a circular-import risk at module load time

        query = input_data["query"]
        holdings = pd.read_csv("data/sample_portfolio.csv")
        prices = pd.read_csv("data/price_history.csv", index_col=0, parse_dates=True)
        held_tickers = holdings["ticker"].tolist()

        requested = self._select_analyses(query, call_llm)
        print(f"[{self.name}] selected analyses: {requested['analyses']} ({requested['reasoning']})")

        metrics = {}

        for ticker in held_tickers:
            ticker_metrics = {}
            if ticker not in prices.columns:
                continue  # price history not available for this ticker — skip rather than crash

            price_series = prices[ticker]

            if "volatility" in requested["analyses"]:
                returns = calculate_returns(price_series)
                ticker_metrics["volatility"] = calculate_volatility(returns)

            if "drawdown" in requested["analyses"]:
                drawdown_series = calculate_drawdown(price_series)
                ticker_metrics["max_drawdown"] = float(drawdown_series.min())

            if "fundamentals" in requested["analyses"]:
                yf_ticker = f"{ticker}.NS"
                ticker_metrics["fundamentals"] = fundamentals_tools.get_fundamentals_summary(yf_ticker)

            if ticker_metrics:
                metrics[ticker] = ticker_metrics

        # Correlation is portfolio-wide, not per-ticker — only compute it
        # if requested AND there's more than one holding to correlate.
        if "correlation" in requested["analyses"] and len(held_tickers) > 1:
            held_prices = prices[held_tickers]
            corr_matrix = calculate_correlation(held_prices)
            metrics["correlation_matrix"] = corr_matrix.to_dict()

        return {
            "agent": self.name,
            "tickers_analyzed": held_tickers,
            "analyses_performed": requested["analyses"],
            "metrics": metrics,
        }

    def _select_analyses(self, query: str, call_llm) -> dict:
        """Same defensive JSON-parsing pattern as judge.score_argument
        and orchestrator.plan_query — strip code fences, validate shape,
        fall back safely on any error."""
        try:
            raw_response = call_llm(system_prompt=TOOL_SELECTION_PROMPT, user_message=query)
            cleaned = raw_response.strip()

            match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
            stripped = match.group(1) if match else cleaned

            parsed = json.loads(stripped)

            if not isinstance(parsed, dict) or "analyses" not in parsed:
                raise ValueError("Missing 'analyses' key")
            if not isinstance(parsed["analyses"], list):
                raise ValueError("'analyses' is not a list")

            valid_analyses = {"volatility", "drawdown", "correlation", "fundamentals"}
            analyses = [a for a in parsed["analyses"] if a in valid_analyses]

            return {
                "analyses": analyses,
                "reasoning": str(parsed.get("reasoning", "")),
            }

        except Exception as e:
            print(f"[{self.name}] tool selection failed, using fallback: {e}")
            return FALLBACK_ANALYSES.copy()


        
from agents.data_analyst import DataAnalystAgent

agent = DataAnalystAgent()

print("--- Query needing fundamentals ---")
result1 = agent.run({"query": "is my portfolio holding any financially risky companies?"})
print(json.dumps(result1, indent=2, default=str))

print("\n--- Query needing only volatility ---")
result2 = agent.run({"query": "what's my portfolio's volatility?"})
print(json.dumps(result2, indent=2, default=str))