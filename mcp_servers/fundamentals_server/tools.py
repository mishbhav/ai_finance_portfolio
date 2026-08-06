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
        # A trailing P/E far above the forward P/E usually signals a
        # recent earnings trough (near-zero trailing earnings inflate
        # the ratio), not genuine overvaluation. Flag rather than hide it.
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
        # yfinance reports this as a percentage (e.g. 280.2 == a D/E
        # ratio of 2.8) — normalized to an actual ratio here so no
        # downstream consumer has to know or guess the raw API's scale.
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


def get_fundamentals_summary(ticker: str) -> dict:
    """Combined fetch: all four categories for one ticker, using a single
    underlying yfinance call. This is the function debate/data-analyst
    agents should actually call — one tool call, full picture — rather
    than four separate ones for the same ticker."""
    info = _get_info(ticker)  # fetched once, reused by all four helpers below

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
    }