import pandas as pd

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Daily % change. Concept: pandas .pct_change()
    turns a price series into a returns series."""

    return prices.pct_change()

def calculate_volatility(returns: pd.Series) -> float:
    """Standard deviation of returns — the standard
    measure of how much a holding swings. Concept: .std()"""

    return float(returns.std())

def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """How far below the running peak the price currently is.
    Concept: running max via .cummax(), then compare current
    price to that peak."""
    running_peak = prices.cummax()
    return (prices-running_peak)/running_peak

def calculate_correlation(price_df: pd.DataFrame) -> pd.DataFrame:
    """How holdings move together. Concept: DataFrame.corr()
    on a wide DataFrame (one column per ticker)."""
    returns_df = price_df.pct_change()
    return returns_df.corr()