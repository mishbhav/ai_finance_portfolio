import yfinance as yf


def _get_info(ticker: str) -> dict:
    """Single fetch point — every function below calls this instead of
    hitting yfinance separately, avoiding redundant network round-trips
    for what is ultimately the same underlying dict per ticker."""
    return yf.Ticker(ticker).info


def get_valuation_metrics(ticker: str) -> dict:
    """Pulls valuation ratios. Returns None for any field yfinance
    doesn't have for this ticker — never raises, never omits a key."""
    info = _get_info(ticker)
    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    return {
        "ticker": ticker,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "price_to_book": info.get("priceToBook"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "trailing_pe_distorted": (
            trailing_pe is not None and forward_pe is not None
            and trailing_pe > forward_pe * 3
        ),
    }


def get_growth_metrics(ticker: str) -> dict:
    """Pulls growth figures (YoY, as reported by yfinance)."""
    info = _get_info(ticker)
    return {
        "ticker": ticker,
        "earnings_growth": info.get("earningsGrowth"),
        "revenue_growth": info.get("revenueGrowth"),
    }


def get_financial_health(ticker: str) -> dict:
    """Pulls balance-sheet health indicators."""
    info = _get_info(ticker)
    debt_to_equity_pct = info.get("debtToEquity")
    return {
        "ticker": ticker,
        "debt_to_equity": debt_to_equity_pct / 100 if debt_to_equity_pct is not None else None,
        "current_ratio": info.get("currentRatio"),
        "return_on_equity": info.get("returnOnEquity"),
    }


def get_dividend_info(ticker: str) -> dict:
    """Pulls dividend yield and payout ratio."""
    info = _get_info(ticker)
    return {
        "ticker": ticker,
        "dividend_yield": info.get("dividendYield"),
        "payout_ratio": info.get("payoutRatio"),
    }


def get_analyst_sentiment(ticker: str) -> dict:
    """Pulls analyst recommendation trend and price targets. Coverage
    varies unpredictably by ticker — don't assume it correlates with
    news or fundamentals coverage for the same name."""
    stock = yf.Ticker(ticker)

    latest_trend = None
    try:
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            latest_row = recommendations.iloc[0]
            latest_trend = {
                "strong_buy": int(latest_row.get("strongBuy", 0)),
                "buy": int(latest_row.get("buy", 0)),
                "hold": int(latest_row.get("hold", 0)),
                "sell": int(latest_row.get("sell", 0)),
                "strong_sell": int(latest_row.get("strongSell", 0)),
            }
    except Exception:
        latest_trend = None

    info = _get_info(ticker)
    current_price = info.get("currentPrice")
    mean_target = info.get("targetMeanPrice")

    implied_upside_pct = None
    if current_price and mean_target:
        implied_upside_pct = round(((mean_target - current_price) / current_price) * 100, 1)

    return {
        "ticker": ticker,
        "recommendation_trend": latest_trend,
        "mean_price_target": mean_target,
        "high_price_target": info.get("targetHighPrice"),
        "low_price_target": info.get("targetLowPrice"),
        "current_price": current_price,
        "implied_upside_pct": implied_upside_pct,
        "num_analyst_opinions": info.get("numberOfAnalystOpinions"),
    }


def get_fundamentals_summary(ticker: str) -> dict:
    """Combined fetch: all five categories for one ticker. This is the
    function debate/data-analyst agents should actually call — one
    logical unit, full picture — rather than calling each piece
    separately. Note: analyst_sentiment makes its own extra network
    call for recommendations (not covered by the shared _get_info
    fetch), so this function is not a single round-trip like the other
    four categories combined would be — a known, acceptable cost for
    bundling everything an agent needs about a ticker in one place."""
    info = _get_info(ticker)

    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    debt_to_equity_pct = info.get("debtToEquity")

    return {
        "ticker": ticker,
        "valuation": {
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "price_to_book": info.get("priceToBook"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "trailing_pe_distorted": (
                trailing_pe is not None and forward_pe is not None
                and trailing_pe > forward_pe * 3
            ),
        },
        "growth": {
            "earnings_growth": info.get("earningsGrowth"),
            "revenue_growth": info.get("revenueGrowth"),
        },
        "financial_health": {
            "debt_to_equity": debt_to_equity_pct / 100 if debt_to_equity_pct is not None else None,
            "current_ratio": info.get("currentRatio"),
            "return_on_equity": info.get("returnOnEquity"),
        },
        "dividends": {
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
        },
        "analyst_sentiment": get_analyst_sentiment(ticker),
    }