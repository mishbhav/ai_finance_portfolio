"""
Fetches real historical daily prices for portfolio holdings via yfinance,
replacing the earlier synthetic random-walk placeholder. Run this whenever
you want to refresh price_history.csv (prices go stale; unlike news, there's
no orchestrator-level auto-refresh for this yet — see note at the bottom).
"""
import yfinance as yf
import pandas as pd


def fetch_price_history(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Fetches adjusted close prices for all tickers into one aligned
    wide DataFrame (one column per ticker, one row per trading date).
    Handles per-ticker fetch failures without crashing the whole batch,
    and forward-fills small gaps rather than leaving holes that would
    silently break downstream returns/correlation calculations."""
    price_data = {}

    for ticker in tickers:
        yf_ticker = f"{ticker}.NS"
        try:
            history = yf.Ticker(yf_ticker).history(period=period, auto_adjust=True)
            if history.empty:
                print(f"[price_history] WARNING: no data returned for {ticker}, skipping")
                continue
            price_data[ticker] = history["Close"]
        except Exception as e:
            print(f"[price_history] WARNING: fetch failed for {ticker}: {e}")
            continue

    if not price_data:
        raise RuntimeError("No price data could be fetched for any ticker — check network/tickers")

    # Combine into one wide DataFrame. Different tickers may have slightly
    # different trading calendars (e.g. one had a data gap on a specific
    # day) — outer join on the date index, then forward-fill small gaps
    # so a single missing day for one ticker doesn't produce a NaN that
    # breaks pct_change()/corr() for the whole row.
    price_df = pd.DataFrame(price_data)
    price_df = price_df.ffill().dropna()

    # yfinance returns tz-aware timestamps; strip the timezone so this
    # matches the plain date index format the rest of the codebase expects
    # (same shape as the old synthetic CSV's index).
    price_df.index = price_df.index.tz_localize(None)

    return price_df


if __name__ == "__main__":
    holdings = pd.read_csv("data/sample_portfolio.csv")
    tickers = holdings["ticker"].unique().tolist()

    price_df = fetch_price_history(tickers)
    price_df.to_csv("data/price_history.csv")

    print(f"Fetched {len(price_df)} trading days for {len(price_df.columns)}/{len(tickers)} tickers")
    missing = set(tickers) - set(price_df.columns)
    if missing:
        print(f"⚠️  No price data for: {sorted(missing)} — these tickers will be excluded from correlation/simulation until resolved")