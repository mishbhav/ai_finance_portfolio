from mcp.server.fastmcp import FastMCP
from tools import calculate_returns, calculate_volatility, calculate_drawdown, calculate_correlation
import pandas as pd

mcp = FastMCP("pandas-analyst")

@mcp.tool()
def get_volatility(prices: list[float]) -> float:
    """Compute volatility for a list of prices."""
    series = pd.Series(prices)
    returns = calculate_returns(series)
    return calculate_volatility(returns)

@mcp.tool()
def get_returns(prices: list[float]) -> list[float]:
    """Compute returns for a list of prices."""
    series = pd.Series(prices)
    returns = calculate_returns(series)
    return returns.astype(object).where(returns.notna(), None).tolist()

@mcp.tool()
def get_drawdown(prices: list[float]) -> list[float]:
    """Compute drawdown for a list of prices."""
    series = pd.Series(prices)
    drawdowns = calculate_drawdown(series)
    return drawdowns.astype(object).where(drawdowns.notna(), None).tolist()

@mcp.tool()
def get_correlation(assets_data: dict[str, list[float]]) -> str:
    """
    Compute a square cross-correlation matrix across multiple tickers.
    
    Expects a dictionary where keys are asset tickers and values are equal-length historical price lists:
    e.g., {"AAPL": [150.0, 152.0, 151.0], "MSFT": [300.0, 305.0, 302.0]}
    Returns a JSON string representing the correlation matrix.
    """
    # Construct a wide DataFrame directly from the uniform ticker map
    price_df = pd.DataFrame(assets_data)
    correlation_matrix = calculate_correlation(price_df)
    
    # Serialize the resulting matrix structure to an easily parsed JSON string payload
    return correlation_matrix.to_json(orient="split")

if __name__ == "__main__":
    mcp.run()