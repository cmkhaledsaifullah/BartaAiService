import logging
from typing import Any

from app.config import get_settings
from app.constants import COLLECTION_NEWS_ARTICLES
from app.database.mongodb import get_collection

logger = logging.getLogger(__name__)

NEWS_COLLECTION = COLLECTION_NEWS_ARTICLES

# RRF constant — standard value used by Elasticsearch, Azure AI Search, etc.
RRF_K = 60

# Fields projected in search results
_RESULT_PROJECTION = {
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
}


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
                **_RESULT_PROJECTION,
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


async def text_search(
    query: str,
    limit: int | None = None,
    pre_filter: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Perform a MongoDB Atlas full-text (BM25) search on the news articles collection.

    Requires a MongoDB Atlas Search index named 'news_text_index' on the
    Title, Body, and Tags fields.

    Args:
        query: The text query to search for.
        limit: Max number of results to return.
        pre_filter: Optional MQL filter to apply after text search.

    Returns:
        List of matching news article documents with text relevance scores.
    """
    settings = get_settings()
    search_limit = limit or settings.vector_search_limit

    search_stage: dict[str, Any] = {
        "index": settings.text_search_index_name,
        "text": {
            "query": query,
            "path": ["Title", "Body", "Tags"],
        },
    }

    pipeline: list[dict[str, Any]] = [
        {"$search": search_stage},
        {
            "$project": {
                **_RESULT_PROJECTION,
                "score": {"$meta": "searchScore"},
            }
        },
        {"$limit": search_limit},
    ]

    # Add post-filter if provided
    if pre_filter:
        pipeline.insert(2, {"$match": pre_filter})

    collection = get_collection(NEWS_COLLECTION)
    results = []
    async for doc in collection.aggregate(pipeline):
        results.append(doc)

    logger.info("Text search returned %d results", len(results))
    return results


def reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = RRF_K,
    final_limit: int = 10,
) -> list[dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank_i)) for each ranker that returns the document.
    This is the same algorithm used by Azure AI Search and Elasticsearch.

    Args:
        *result_lists: One or more ordered result lists (best-first).
        k: RRF constant (default 60). Higher values reduce the impact of rank position.
        final_limit: Number of results to return after fusion.

    Returns:
        Merged and re-ranked list of documents with an 'rrf_score' field.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            news_id = doc.get("NewsId", "")
            if not news_id:
                continue
            rrf_scores[news_id] = rrf_scores.get(news_id, 0.0) + 1.0 / (k + rank)
            # Keep the first occurrence (highest individual score) for each doc
            if news_id not in doc_map:
                doc_map[news_id] = doc

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    results = []
    for news_id in sorted_ids[:final_limit]:
        doc = dict(doc_map[news_id])
        # Replace individual score with the combined RRF score
        doc["score"] = round(rrf_scores[news_id], 6)
        results.append(doc)

    return results


async def hybrid_search(
    query: str,
    query_embedding: list[float],
    limit: int | None = None,
    pre_filter: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Perform hybrid search combining vector (semantic) and text (BM25) results
    using Reciprocal Rank Fusion.

    Runs both searches, then merges and re-ranks with RRF.

    Args:
        query: The raw text query (for BM25 search).
        query_embedding: The embedding vector (for vector search).
        limit: Max number of final results to return.
        pre_filter: Optional MQL filter applied to both searches.

    Returns:
        Merged, re-ranked list of documents.
    """
    settings = get_settings()
    search_limit = limit or settings.vector_search_limit
    # Fetch more candidates from each ranker so RRF has better coverage
    candidate_limit = search_limit * 3

    vector_results = await vector_search(
        query_embedding=query_embedding,
        limit=candidate_limit,
        pre_filter=pre_filter,
    )
    text_results = await text_search(
        query=query,
        limit=candidate_limit,
        pre_filter=pre_filter,
    )

    merged = reciprocal_rank_fusion(
        vector_results, text_results, final_limit=search_limit
    )
    logger.info(
        "Hybrid search: %d vector + %d text → %d merged results",
        len(vector_results),
        len(text_results),
        len(merged),
    )
    return merged


async def ensure_indexes() -> None:
    """Ensure required database indexes exist (non-vector indexes)."""
    collection = get_collection(NEWS_COLLECTION)
    await collection.create_index("NewsId", unique=True)
    await collection.create_index("CategoryId")
    await collection.create_index("NewsPaperId")
    await collection.create_index("PublishDate")
    await collection.create_index("Tags")
    logger.info("Database indexes ensured for %s", NEWS_COLLECTION)
