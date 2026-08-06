"""
Standalone, read-only diagnostic for yfinance news data quality.
Does NOT touch chroma_db/, does NOT modify the orchestrator.
Run this manually whenever you want to sanity-check news coverage
for your actual holdings.
"""
import yfinance as yf
import pandas as pd
import json

# All current holdings are NSE-listed.
EXCHANGE_SUFFIX = ".NS"


def diagnose_ticker(ticker: str) -> dict:
    """Fetches news for one ticker and reports on data quality —
    does not assume any particular field schema going in."""
    yf_ticker = f"{ticker}{EXCHANGE_SUFFIX}"
    result = {
        "ticker": ticker,
        "yf_ticker": yf_ticker,
        "article_count": 0,
        "quality": "empty",
        "sample_raw": None,
        "error": None,
    }

    try:
        articles = yf.Ticker(yf_ticker).news
        result["article_count"] = len(articles)

        if not articles:
            return result  # quality stays "empty"

        result["sample_raw"] = articles[0]

        # Real fields live nested under "content", not at the top level
        content = articles[0].get("content", {})
        summary = content.get("summary", "")
        title = content.get("title", "")

        result["quality"] = "good" if len(summary) > 40 else ("thin" if title else "empty")

    except Exception as e:
        result["error"] = str(e)
        result["quality"] = "error"

    return result


def validate_ticker_exists(ticker: str) -> dict:
    """For tickers with zero news articles, checks whether the ticker
    symbol itself is valid/tradeable — distinguishes 'real ticker, just
    quiet on news' from 'wrong ticker symbol entirely'."""
    yf_ticker = f"{ticker}{EXCHANGE_SUFFIX}"
    result = {"ticker": ticker, "yf_ticker": yf_ticker, "valid": False, "last_price": None, "error": None}

    try:
        info = yf.Ticker(yf_ticker).fast_info
        price = info.get("lastPrice")
        if price:
            result["valid"] = True
            result["last_price"] = price
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    holdings = pd.read_csv("data/sample_portfolio.csv")
    tickers = holdings["ticker"].tolist()

    print(f"Diagnosing {len(tickers)} tickers: {tickers}\n")

    results = [diagnose_ticker(t) for t in tickers]

    # Summary table
    print(f"{'Ticker':<12}{'YF Symbol':<15}{'Articles':<10}{'Quality':<10}")
    print("-" * 47)
    for r in results:
        print(f"{r['ticker']:<12}{r['yf_ticker']:<15}{r['article_count']:<10}{r['quality']:<10}")

    # One full raw sample, from the first ticker that actually returned something
    sample = next((r for r in results if r["sample_raw"]), None)
    if sample:
        print(f"\n--- Raw sample from {sample['ticker']} ---")
        print(json.dumps(sample["sample_raw"], indent=2, default=str))
    else:
        print("\nNo ticker returned any articles — nothing to sample.")

    zero_article_tickers = [r["ticker"] for r in results if r["article_count"] == 0]
    error_tickers = [r["ticker"] for r in results if r["error"]]

    if error_tickers:
        print(f"\n❌ Errors during news fetch for: {error_tickers} — check ticker symbols are valid/resolvable.")

    # Dynamically follow up on every zero-article ticker found above —
    # no manual copy-pasting of the list required.
    if zero_article_tickers:
        print(f"\n⚠️  Zero articles for: {zero_article_tickers}")
        print("Checking whether these tickers are valid/tradeable symbols...\n")

        validations = [validate_ticker_exists(t) for t in zero_article_tickers]

        print(f"{'Ticker':<12}{'YF Symbol':<15}{'Valid?':<10}{'Last Price':<12}")
        print("-" * 49)
        for v in validations:
            valid_label = "yes" if v["valid"] else "NO"
            price_label = f"{v['last_price']}" if v["last_price"] else (v["error"] or "no data")
            print(f"{v['ticker']:<12}{v['yf_ticker']:<15}{valid_label:<10}{price_label:<12}")

        invalid = [v["ticker"] for v in validations if not v["valid"]]
        if invalid:
            print(f"\n❌ Likely wrong ticker symbols (re-verify against ISIN): {invalid}")
        else:
            print("\n✅ All zero-article tickers are valid, tradeable symbols — genuinely low news coverage, not a bad ticker.")


if __name__ == "__main__":
    main()