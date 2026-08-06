from mcp.server.fastmcp import FastMCP
from tools import (
    get_valuation_metrics,
    get_growth_metrics,
    get_financial_health,
    get_dividend_info,
    get_fundamentals_summary,
)

mcp = FastMCP("fundamentals-analyst")


@mcp.tool()
def get_fundamentals(ticker: str) -> dict:
    """Get a full fundamentals picture (valuation, growth, financial
    health, dividends) for an NSE ticker (e.g. 'ASIANPAINT.NS'). Prefer
    this over the individual metric tools below unless you specifically
    need just one category. Note: debt_to_equity is already normalized
    to a true ratio (not a percentage). If trailing_pe_distorted is true,
    trailing P/E is unreliable (likely a recent earnings trough) — weigh
    forward_pe more heavily in that case. Some fields (e.g. current_ratio)
    are commonly null for financial-sector companies (banks, NBFCs) where
    the concept doesn't apply — this is expected, not missing data."""
    return get_fundamentals_summary(ticker)


@mcp.tool()
def get_valuation(ticker: str) -> dict:
    """Get valuation ratios (P/E, P/B, EV/EBITDA) for an NSE ticker."""
    return get_valuation_metrics(ticker)


@mcp.tool()
def get_growth(ticker: str) -> dict:
    """Get earnings/revenue growth figures for an NSE ticker."""
    return get_growth_metrics(ticker)


@mcp.tool()
def get_financial_health_tool(ticker: str) -> dict:
    """Get debt/liquidity/profitability ratios for an NSE ticker.
    debt_to_equity is normalized to a true ratio, not a percentage."""
    return get_financial_health(ticker)


@mcp.tool()
def get_dividends(ticker: str) -> dict:
    """Get dividend yield and payout ratio for an NSE ticker."""
    return get_dividend_info(ticker)


if __name__ == "__main__":
    mcp.run()