import logging
from typing import Any

from app.config import get_settings
from app.constants import COLLECTION_NEWS_ARTICLES
from app.database.mongodb import get_collection

logger = logging.getLogger(__name__)

NEWS_COLLECTION = COLLECTION_NEWS_ARTICLES


async def vector_search(
    query_embedding: list[float],
    limit: int | None = None,
    pre_filter: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Perform a MongoDB Atlas Vector Search on the news articles collection.

    Args:
        query_embedding: The embedding vector to search against.
        limit: Max number of results to return.
        pre_filter: Optional MQL filter to apply before vector search.

    Returns:
        List of matching news article documents with relevance scores.
    """
    settings = get_settings()
    num_candidates = settings.vector_search_num_candidates
    search_limit = limit or settings.vector_search_limit

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": settings.vector_search_index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": search_limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "NewsId": 1,
                "NewsPaperId": 1,
                "CategoryId": 1,
                "Title": 1,
                "Body": 1,
                "Tags": 1,
                "PublishDate": 1,
                "Author": 1,
                "SourceURL": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    # Add pre-filter if provided
    if pre_filter:
        pipeline[0]["$vectorSearch"]["filter"] = pre_filter

    collection = get_collection(NEWS_COLLECTION)
    results = []
    async for doc in collection.aggregate(pipeline):
        results.append(doc)

    logger.info("Vector search returned %d results", len(results))
    return results


async def ensure_indexes() -> None:
    """Ensure required database indexes exist (non-vector indexes)."""
    collection = get_collection(NEWS_COLLECTION)
    await collection.create_index("NewsId", unique=True)
    await collection.create_index("CategoryId")
    await collection.create_index("NewsPaperId")
    await collection.create_index("PublishDate")
    await collection.create_index("Tags")
    logger.info("Database indexes ensured for %s", NEWS_COLLECTION)
