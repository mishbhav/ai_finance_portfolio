"""
Run manually against real holdings to sanity-check actual data availability —
not a pytest test, a review aid, same role as manual_eval_debate.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp_servers" / "fundamentals_server"))

from tools import get_fundamentals_summary
import pandas as pd
import json

holdings = pd.read_csv("data/sample_portfolio.csv")
tickers = holdings["ticker"].tolist()

for ticker in tickers:
    yf_ticker = f"{ticker}.NS"
    print(f"\n{'='*50}\n{ticker}\n{'='*50}")
    print(json.dumps(get_fundamentals_summary(yf_ticker), indent=2, default=str))