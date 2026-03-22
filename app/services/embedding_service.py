"""
Pluggable embedding service.

Supports multiple embedding providers configured via EMBEDDING_PROVIDER in .env:
  - openai        : OpenAI text-embedding-3-small/large (API)
  - cohere        : Cohere embed-multilingual-v3.0 (API, excellent Bangla)
  - local         : sentence-transformers models run locally (free, no API key)
"""

import asyncio
import logging
from abc import ABC, abstractmethod

from app.config import get_settings
from app.constants import ERROR_UNKNOWN_EMBEDDING_PROVIDER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Base class for all embedding providers."""

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API embedding provider (text-embedding-3-small / large)."""

    DIMENSIONS_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self):
        from openai import AsyncOpenAI

        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dims = self.DIMENSIONS_MAP.get(self._model, 1536)

    @property
    def dimensions(self) -> int:
        return self._dims

    async def generate_embedding(self, text: str) -> list[float]:
        truncated = text[:30000]
        response = await self._client.embeddings.create(
            input=truncated, model=self._model
        )
        return response.data[0].embedding

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        truncated = [t[:30000] for t in texts]
        response = await self._client.embeddings.create(
            input=truncated, model=self._model
        )
        return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Cohere provider
# ---------------------------------------------------------------------------

class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere API embedding provider (embed-multilingual-v3.0, excellent for Bangla)."""

    DIMENSIONS_MAP = {
        "embed-multilingual-v3.0": 1024,
        "embed-multilingual-light-v3.0": 384,
        "embed-english-v3.0": 1024,
    }

    def __init__(self):
        import cohere

        settings = get_settings()
        self._client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
        self._model = settings.embedding_model
        self._dims = self.DIMENSIONS_MAP.get(self._model, 1024)

    @property
    def dimensions(self) -> int:
        return self._dims

    async def generate_embedding(self, text: str) -> list[float]:
        truncated = text[:10000]
        response = await self._client.embed(
            texts=[truncated],
            model=self._model,
            input_type="search_document",
            embedding_types=["float"],
        )
        return response.embeddings.float_[0]

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        truncated = [t[:10000] for t in texts]
        # Cohere supports up to 96 texts per call
        all_embeddings: list[list[float]] = []
        for i in range(0, len(truncated), 96):
            batch = truncated[i : i + 96]
            response = await self._client.embed(
                texts=batch,
                model=self._model,
                input_type="search_document",
                embedding_types=["float"],
            )
            all_embeddings.extend(response.embeddings.float_)
        return all_embeddings


# ---------------------------------------------------------------------------
# Local (sentence-transformers) provider
# ---------------------------------------------------------------------------

class LocalEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embedding provider (free, no API key required)."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self._model_name = settings.embedding_model
        logger.info("Loading local embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._dims = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Local model loaded: %s (dimensions: %d)", self._model_name, self._dims
        )

    @property
    def dimensions(self) -> int:
        return self._dims

    async def generate_embedding(self, text: str) -> list[float]:
        truncated = text[:10000]
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self._model.encode(truncated).tolist()
        )
        return embedding

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        truncated = [t[:10000] for t in texts]
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(truncated).tolist()
        )
        return embeddings


# ---------------------------------------------------------------------------
# Provider factory & public API
# ---------------------------------------------------------------------------

_provider: EmbeddingProvider | None = None

PROVIDERS = {
    "openai": OpenAIEmbeddingProvider,
    "cohere": CohereEmbeddingProvider,
    "local": LocalEmbeddingProvider,
}


def get_embedding_provider() -> EmbeddingProvider:
    """Get or create the configured embedding provider (singleton)."""
    global _provider
    if _provider is None:
        settings = get_settings()
        provider_name = settings.embedding_provider.lower()
        provider_class = PROVIDERS.get(provider_name)
        if provider_class is None:
            raise ValueError(
                ERROR_UNKNOWN_EMBEDDING_PROVIDER.format(
                    provider=provider_name,
                    supported=", ".join(PROVIDERS.keys()),
                )
            )
        _provider = provider_class()
        logger.info(
            "Embedding provider initialized: %s (model: %s, dimensions: %d)",
            provider_name,
            settings.embedding_model,
            _provider.dimensions,
        )
    return _provider


async def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector using the configured provider."""
    return await get_embedding_provider().generate_embedding(text)


async def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts using the configured provider."""
    return await get_embedding_provider().generate_embeddings_batch(texts)


def get_embedding_dimensions() -> int:
    """Return the dimensionality of the current embedding model."""
    return get_embedding_provider().dimensions
