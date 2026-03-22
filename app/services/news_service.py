import logging
from datetime import datetime, timezone
from typing import Any

from app.constants import COLLECTION_CLICK_LOGS
from app.database.mongodb import get_collection
from app.database.vector_store import NEWS_COLLECTION

logger = logging.getLogger(__name__)


async def get_article_by_id(news_id: str) -> dict | None:
    """Retrieve a single news article by its NewsId."""
    collection = get_collection(NEWS_COLLECTION)
    return await collection.find_one(
        {"NewsId": news_id}, {"_id": 0, "embedding": 0}
    )


async def search_articles_by_filter(
    filters: dict[str, Any],
    limit: int = 20,
    sort_by: str = "PublishDate",
    sort_order: int = -1,
) -> list[dict]:
    """
    Search articles using MongoDB query filters (non-vector).

    Useful for filtering by category, newspaper, date range, etc.
    """
    collection = get_collection(NEWS_COLLECTION)
    cursor = (
        collection.find(filters, {"_id": 0, "embedding": 0})
        .sort(sort_by, sort_order)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)


async def get_recent_articles(
    limit: int = 10,
    category_id: str | None = None,
    newspaper_id: str | None = None,
) -> list[dict]:
    """Get the most recent articles, optionally filtered by category or newspaper."""
    filters: dict[str, Any] = {}
    if category_id:
        filters["CategoryId"] = category_id
    if newspaper_id:
        filters["NewsPaperId"] = newspaper_id

    return await search_articles_by_filter(filters, limit=limit)


async def get_articles_by_tags(tags: list[str], limit: int = 10) -> list[dict]:
    """Find articles matching any of the given tags."""
    filters = {"Tags": {"$in": tags}}
    return await search_articles_by_filter(filters, limit=limit)


async def get_article_by_source_url(source_url: str) -> dict | None:
    """Look up an article by its SourceURL field."""
    collection = get_collection(NEWS_COLLECTION)
    return await collection.find_one(
        {"SourceURL": source_url}, {"_id": 0, "embedding": 0}
    )


async def log_click(query: str, news_id: str, source_url: str = "") -> None:
    """Persist a query-article click pair for future training data.

    Each record stores the query, the resolved NewsId, and metadata.
    These pairs will later be used for self-supervised embedding fine-tuning.
    """
    collection = get_collection(COLLECTION_CLICK_LOGS)
    await collection.insert_one(
        {
            "query": query,
            "news_id": news_id,
            "source_url": source_url,
            "clicked_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    logger.info("Click logged: query=%s news_id=%s", query[:80], news_id)


async def ensure_click_log_indexes() -> None:
    """Create indexes for the click_logs collection."""
    collection = get_collection(COLLECTION_CLICK_LOGS)
    await collection.create_index("news_id")
    await collection.create_index("clicked_at")
    logger.info("Indexes ensured for %s", COLLECTION_CLICK_LOGS)
