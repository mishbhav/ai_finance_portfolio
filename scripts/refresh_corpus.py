import yfinance as yf
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PERSIST_DIR = "chroma_db"
STATE_PATH = Path(__file__).resolve().parent.parent / "memory" / "refresh_state.json"

_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)
_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def fetch_news_for_ticker(ticker: str) -> list[dict]:
    """Fetches recent news for one NSE ticker, returning a clean,
    consistent shape regardless of what yfinance's raw response looks
    like. Never raises — a fetch problem for one ticker shouldn't be
    the caller's problem to handle article-by-article."""
    yf_ticker = f"{ticker}.NS"
    articles = []

    try:
        raw_items = yf.Ticker(yf_ticker).news
    except Exception as e:
        print(f"[refresh_corpus] fetch failed for {ticker}: {e}")
        return []

    for item in raw_items:
        content = item.get("content", {})
        title = content.get("title", "")
        summary = content.get("summary", "")

        if not summary:
            continue  # no usable evidence text — skip, don't index an empty chunk

        provider = content.get("provider", {})
        articles.append({
            "ticker": ticker,
            "title": title,
            "summary": summary,
            "published": content.get("pubDate", ""),
            "source": provider.get("displayName", "unknown"),
            "type": "news",
        })

    return articles


def _make_chunk_id(ticker: str, text: str) -> str:
    """Deterministic ID from ticker + chunk content — the same article
    chunk fetched twice produces the same ID, so re-adding it upserts
    rather than duplicates."""
    raw = f"{ticker}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _articles_to_chunks(articles: list[dict]) -> tuple[list[Document], list[str]]:
    """Converts fetched articles into chunked Documents + matching dedup IDs."""
    documents = []
    for article in articles:
        full_text = f"{article['title']}\n\n{article['summary']}"
        documents.append(Document(
            page_content=full_text,
            metadata={
                "ticker": article["ticker"],
                "source": article["source"],
                "published": article["published"],
                "type": article["type"],
            },
        ))

    chunks = _splitter.split_documents(documents)
    ids = [_make_chunk_id(chunk.metadata["ticker"], chunk.page_content) for chunk in chunks]
    return chunks, ids


def _should_skip_refresh(min_minutes: int = 60) -> bool:
    """Avoids redundant fetches if the corpus was refreshed recently.
    Returns True if we should skip (i.e. last refresh was < min_minutes ago).
    Fails toward NOT skipping (returns False) on any read/parse error —
    same philosophy as plan_query's fallback to full_debate."""
    if not STATE_PATH.exists():
        return False

    try:
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
        last_refresh = datetime.fromisoformat(state["last_refresh"])
        elapsed = datetime.now(timezone.utc) - last_refresh
        return elapsed.total_seconds() < (min_minutes * 60)
    except Exception as e:
        print(f"[refresh_corpus] couldn't read refresh state, will refresh: {e}")
        return False


def _record_refresh_time() -> None:
    STATE_PATH.parent.mkdir(exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({"last_refresh": datetime.now(timezone.utc).isoformat()}, f)


def refresh(tickers: list[str], force: bool = False) -> int:
    """Fetches recent news for each ticker and upserts it into the
    persisted Chroma store. Skips entirely if refreshed recently, unless
    force=True. Returns the number of chunks added (0 if skipped)."""
    if not force and _should_skip_refresh():
        print("[refresh_corpus] skipped — refreshed recently")
        return 0

    all_articles = []
    for ticker in tickers:
        articles = fetch_news_for_ticker(ticker)
        all_articles.extend(articles)

    if not all_articles:
        print("[refresh_corpus] no articles fetched — corpus unchanged")
        _record_refresh_time()
        return 0

    chunks, ids = _articles_to_chunks(all_articles)
    _vector_store.add_documents(chunks, ids=ids)
    _record_refresh_time()

    print(f"[refresh_corpus] added {len(chunks)} chunks from {len(all_articles)} articles across {len(tickers)} tickers")
    return len(chunks)


if __name__ == "__main__":
    import pandas as pd
    holdings = pd.read_csv("data/sample_portfolio.csv")
    refresh(holdings["ticker"].tolist())