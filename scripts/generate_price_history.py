import pandas as pd
import numpy as np

def generate_price_history(tickers: list[str], days: int = 252, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily prices via random walk — placeholder until a real
    market-data source is wired in for price history specifically (fundamentals
    and news already use real yfinance data; only historical daily prices
    for volatility/drawdown/correlation/simulation are still synthetic).
    Each ticker gets independent random drift/volatility so correlation
    calculations aren't trivially 1.0."""
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)

    data = {}
    for ticker in tickers:
        daily_returns = np.random.normal(loc=0.0004, scale=0.015, size=days)
        prices = 100 * (1 + daily_returns).cumprod()
        data[ticker] = prices

    return pd.DataFrame(data, index=dates)

if __name__ == "__main__":
    holdings = pd.read_csv("data/sample_portfolio.csv")
    tickers = holdings["ticker"].unique().tolist()

    price_df = generate_price_history(tickers)
    price_df.to_csv("data/price_history.csv")
    print(f"Generated {len(price_df)} days of price history for: {tickers}")