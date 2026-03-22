"""
Script to index existing news articles with vector embeddings.

This script reads all news articles from MongoDB that don't have
embeddings yet and generates embeddings using OpenAI, then updates
the documents in-place.

Usage:
    python -m scripts.index_articles

Environment:
    Requires .env file with MONGODB_URI and OPENAI_API_KEY set.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient
from app.config import get_settings
from app.constants import COLLECTION_NEWS_ARTICLES
from app.services.embedding_service import generate_embeddings_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 50
NEWS_COLLECTION = COLLECTION_NEWS_ARTICLES


def build_embedding_text(article: dict) -> str:
    """Build the text to embed from an article's fields."""
    parts = [
        article.get("Title", ""),
        article.get("Body", ""),
    ]
    tags = article.get("Tags", [])
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    author = article.get("Author", "")
    if author:
        parts.append(f"Author: {author}")
    return "\n".join(parts)


async def index_articles():
    settings = get_settings()
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_db_name]
    collection = db[NEWS_COLLECTION]

    # Count articles without embeddings
    total = await collection.count_documents({"embedding": {"$exists": False}})
    logger.info("Found %d articles without embeddings", total)

    if total == 0:
        logger.info("All articles already indexed. Nothing to do.")
        return

    processed = 0
    cursor = collection.find(
        {"embedding": {"$exists": False}},
        {"NewsId": 1, "Title": 1, "Body": 1, "Tags": 1, "Author": 1},
    )

    batch_docs = []
    batch_texts = []

    async for doc in cursor:
        text = build_embedding_text(doc)
        batch_docs.append(doc)
        batch_texts.append(text)

        if len(batch_docs) >= BATCH_SIZE:
            await _process_batch(collection, batch_docs, batch_texts)
            processed += len(batch_docs)
            logger.info("Progress: %d / %d", processed, total)
            batch_docs = []
            batch_texts = []

    # Process remaining
    if batch_docs:
        await _process_batch(collection, batch_docs, batch_texts)
        processed += len(batch_docs)

    logger.info("Indexing complete. Processed %d articles.", processed)
    client.close()


async def _process_batch(collection, docs: list, texts: list):
    """Generate embeddings for a batch and update MongoDB."""
    embeddings = await generate_embeddings_batch(texts)
    operations = []
    from pymongo import UpdateOne

    for doc, embedding in zip(docs, embeddings):
        operations.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {"embedding": embedding}},
            )
        )
    if operations:
        result = await collection.bulk_write(operations)
        logger.info("Batch updated: %d documents", result.modified_count)


if __name__ == "__main__":
    asyncio.run(index_articles())
