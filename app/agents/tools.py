"""
Agent tools for the Agentic RAG pipeline.

Each tool is a function that the LangChain agent can invoke to retrieve
or filter news articles from the database.
"""

import json
import logging
from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from app.constants import (
    DATE_FORMAT_YMD,
    MSG_INVALID_DATE_FORMAT,
    MSG_NO_ARTICLES_BY_CATEGORY,
    MSG_NO_ARTICLES_BY_DATE_RANGE,
    MSG_NO_ARTICLES_BY_NEWSPAPER,
    MSG_NO_ARTICLES_BY_TAGS,
    MSG_NO_ARTICLES_FOUND,
    MSG_NO_RECENT_ARTICLES,
    MSG_NO_TAGS_PROVIDED,
)
from app.database.vector_store import hybrid_search
from app.services.embedding_service import generate_embedding
from app.services.news_service import (
    get_articles_by_tags,
    get_recent_articles,
    search_articles_by_filter,
)

logger = logging.getLogger(__name__)


@tool
async def semantic_news_search(query: str) -> str:
    """Search for Bangladesh news articles semantically related to the query.

    Use this tool when the user asks about a topic and you need to find
    relevant news articles. The search uses hybrid retrieval: vector
    embeddings for semantic similarity combined with BM25 keyword matching,
    merged via Reciprocal Rank Fusion for best results.

    Args:
        query: A natural language description of the news topic to search for.
    """
    embedding = await generate_embedding(query)
    results = await hybrid_search(query=query, query_embedding=embedding)
    if not results:
        return MSG_NO_ARTICLES_FOUND
    return _format_results(results)


@tool
async def search_news_by_category(category_id: str, limit: int = 10) -> str:
    """Search for news articles by category.

    Use this tool when the user asks about a specific news category
    (e.g., politics, sports, business, entertainment, technology).

    Args:
        category_id: The category identifier to filter by.
        limit: Maximum number of articles to return (default 10).
    """
    capped_limit = min(limit, 20)
    results = await get_recent_articles(limit=capped_limit, category_id=category_id)
    if not results:
        return MSG_NO_ARTICLES_BY_CATEGORY.format(category_id=category_id)
    return _format_results(results)


@tool
async def search_news_by_date_range(
    start_date: str, end_date: str, category_id: str = ""
) -> str:
    """Search for news articles published within a date range.

    Use this tool when the user asks about news from a specific time period.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        category_id: Optional category filter.
    """
    try:
        start = datetime.strptime(start_date, DATE_FORMAT_YMD)
        end = datetime.strptime(end_date, DATE_FORMAT_YMD)
    except ValueError:
        return MSG_INVALID_DATE_FORMAT

    filters: dict[str, Any] = {
        "PublishDate": {"$gte": start.isoformat(), "$lte": end.isoformat()}
    }
    if category_id:
        filters["Category"] = category_id

    results = await search_articles_by_filter(filters, limit=15)
    if not results:
        return MSG_NO_ARTICLES_BY_DATE_RANGE.format(start_date=start_date, end_date=end_date)
    return _format_results(results)


@tool
async def search_news_by_tags(tags: str) -> str:
    """Search for news articles matching specific tags.

    Use this tool when the user mentions specific topics, people, places,
    or events that could be article tags.

    Args:
        tags: Comma-separated list of tags to search for.
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if not tag_list:
        return MSG_NO_TAGS_PROVIDED
    results = await get_articles_by_tags(tag_list, limit=10)
    if not results:
        return MSG_NO_ARTICLES_BY_TAGS.format(tags=tags)
    return _format_results(results)


@tool
async def search_news_by_newspaper(newspaper_id: str, limit: int = 10) -> str:
    """Get recent articles from a specific newspaper.

    Use this tool when the user asks about news from a particular
    newspaper or publication.

    Args:
        newspaper_id: The newspaper identifier.
        limit: Maximum number of articles to return (default 10).
    """
    capped_limit = min(limit, 20)
    results = await get_recent_articles(limit=capped_limit, newspaper_id=newspaper_id)
    if not results:
        return MSG_NO_ARTICLES_BY_NEWSPAPER.format(newspaper_id=newspaper_id)
    return _format_results(results)


@tool
async def get_latest_news(limit: int = 5) -> str:
    """Get the most recent news articles.

    Use this tool when the user asks about the latest or most recent news
    without a specific topic.

    Args:
        limit: Number of recent articles to fetch (default 5).
    """
    capped_limit = min(limit, 15)
    results = await get_recent_articles(limit=capped_limit)
    if not results:
        return MSG_NO_RECENT_ARTICLES
    return _format_results(results)


def _format_results(articles: list[dict]) -> str:
    """Format article results into a structured string for the agent."""
    formatted = []
    for i, article in enumerate(articles, 1):
        entry = {
            "index": i,
            "title": article.get("Title", "N/A"),
            "published": article.get("PublishDate", "N/A"),
            "author": article.get("Author", "N/A"),
            "source_url": article.get("SourceURL", "N/A"),
            "category": article.get("Category", "N/A"),
            "newspaper": article.get("NewsPaper", "N/A"),
            "tags": article.get("Tags", []),
            "body": article.get("Body", "N/A")[:2000],  # Truncate long bodies
        }
        if "score" in article:
            entry["relevance_score"] = round(article["score"], 4)
        formatted.append(entry)
    return json.dumps(formatted, ensure_ascii=False, default=str)


# All tools the agent can use
ALL_TOOLS = [
    semantic_news_search,
    search_news_by_category,
    search_news_by_date_range,
    search_news_by_tags,
    search_news_by_newspaper,
    get_latest_news,
]
