import pytest
from unittest.mock import patch, MagicMock, call

from app.services.embedding_service import (
    get_embedding_provider,
    generate_embedding,
    generate_embeddings_batch,
    get_embedding_dimensions,
    OpenAIEmbeddingProvider,
    CohereEmbeddingProvider,
    LocalEmbeddingProvider,
    PROVIDERS,
)
import app.services.embedding_service as embedding_module


class TestProviderFactory:
    def teardown_method(self):
        # Reset singleton between tests
        embedding_module._provider = None

    @patch("app.services.embedding_service.get_settings")
    def test_creates_openai_provider(self, mock_settings):
        mock_settings.return_value = MagicMock(
            embedding_provider="openai", embedding_model="text-embedding-3-small",
            openai_api_key="test-key",
        )
        mock_instance = MagicMock(dimensions=1536)
        with patch.dict(PROVIDERS, {"openai": lambda: mock_instance}):
            provider = get_embedding_provider()
        assert provider is mock_instance

    @patch("app.services.embedding_service.get_settings")
    def test_creates_cohere_provider(self, mock_settings):
        mock_settings.return_value = MagicMock(
            embedding_provider="cohere", embedding_model="embed-multilingual-v3.0",
            cohere_api_key="test-key",
        )
        mock_instance = MagicMock(dimensions=1024)
        with patch.dict(PROVIDERS, {"cohere": lambda: mock_instance}):
            provider = get_embedding_provider()
        assert provider is mock_instance

    @patch("app.services.embedding_service.get_settings")
    def test_unknown_provider_raises(self, mock_settings):
        mock_settings.return_value = MagicMock(
            embedding_provider="invalid", embedding_model="x"
        )
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider()

    @patch("app.services.embedding_service.get_settings")
    def test_singleton_returns_same_instance(self, mock_settings):
        mock_settings.return_value = MagicMock(
            embedding_provider="openai", embedding_model="text-embedding-3-small",
            openai_api_key="test-key",
        )
        mock_factory = MagicMock()
        mock_instance = MagicMock(dimensions=1536)
        mock_factory.return_value = mock_instance
        with patch.dict(PROVIDERS, {"openai": mock_factory}):
            p1 = get_embedding_provider()
            p2 = get_embedding_provider()
        assert p1 is p2
        assert mock_factory.call_count == 1


class TestOpenAIEmbeddingProvider:
    def test_dimensions_map(self):
        dims = OpenAIEmbeddingProvider.DIMENSIONS_MAP
        assert dims["text-embedding-3-small"] == 1536
        assert dims["text-embedding-3-large"] == 3072
        assert dims["text-embedding-ada-002"] == 1536


class TestCohereEmbeddingProvider:
    def test_dimensions_map(self):
        dims = CohereEmbeddingProvider.DIMENSIONS_MAP
        assert dims["embed-multilingual-v3.0"] == 1024
        assert dims["embed-multilingual-light-v3.0"] == 384


class TestPublicApiFunctions:
    def teardown_method(self):
        embedding_module._provider = None

    @pytest.mark.asyncio
    async def test_generate_embedding_delegates(self):
        mock_provider = MagicMock()
        mock_provider.generate_embedding = MagicMock(return_value=[0.1, 0.2])
        # Make it awaitable
        import asyncio
        future = asyncio.Future()
        future.set_result([0.1, 0.2])
        mock_provider.generate_embedding.return_value = future

        embedding_module._provider = mock_provider
        result = await generate_embedding("test text")
        assert result == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_delegates(self):
        mock_provider = MagicMock()
        import asyncio
        future = asyncio.Future()
        future.set_result([[0.1], [0.2]])
        mock_provider.generate_embeddings_batch.return_value = future

        embedding_module._provider = mock_provider
        result = await generate_embeddings_batch(["a", "b"])
        assert result == [[0.1], [0.2]]

    def test_get_embedding_dimensions_delegates(self):
        mock_provider = MagicMock(dimensions=1536)
        embedding_module._provider = mock_provider
        assert get_embedding_dimensions() == 1536


class TestProvidersRegistry:
    def test_all_providers_registered(self):
        assert "openai" in PROVIDERS
        assert "cohere" in PROVIDERS
        assert "local" in PROVIDERS
        assert len(PROVIDERS) == 3


class TestOpenAIEmbeddingProviderInit:
    @patch("app.services.embedding_service.get_settings")
    def test_init_and_dimensions(self, mock_settings):
        mock_settings.return_value = MagicMock(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
        )
        with patch("app.services.embedding_service.OpenAIEmbeddingProvider.__init__", lambda self: None):
            provider = OpenAIEmbeddingProvider()
        # Test DIMENSIONS_MAP directly
        assert OpenAIEmbeddingProvider.DIMENSIONS_MAP["text-embedding-3-small"] == 1536

    @patch("app.services.embedding_service.get_settings")
    @patch("openai.AsyncOpenAI")
    def test_constructor_sets_fields(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
        )
        provider = OpenAIEmbeddingProvider()
        assert provider.dimensions == 1536
        assert provider._model == "text-embedding-3-small"

    @pytest.mark.asyncio
    @patch("app.services.embedding_service.get_settings")
    @patch("openai.AsyncOpenAI")
    async def test_generate_embedding(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        import asyncio
        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.embeddings.create.return_value = future

        provider = OpenAIEmbeddingProvider()
        result = await provider.generate_embedding("test text")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    @patch("app.services.embedding_service.get_settings")
    @patch("openai.AsyncOpenAI")
    async def test_generate_embeddings_batch(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1]),
            MagicMock(embedding=[0.2]),
        ]

        import asyncio
        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.embeddings.create.return_value = future

        provider = OpenAIEmbeddingProvider()
        result = await provider.generate_embeddings_batch(["a", "b"])
        assert result == [[0.1], [0.2]]


class TestCohereEmbeddingProviderInit:
    @patch("app.services.embedding_service.get_settings")
    @patch("cohere.AsyncClientV2")
    def test_constructor_sets_fields(self, mock_cohere_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            cohere_api_key="test-key",
            embedding_model="embed-multilingual-v3.0",
        )
        provider = CohereEmbeddingProvider()
        assert provider.dimensions == 1024
        assert provider._model == "embed-multilingual-v3.0"

    @pytest.mark.asyncio
    @patch("app.services.embedding_service.get_settings")
    @patch("cohere.AsyncClientV2")
    async def test_generate_embedding(self, mock_cohere_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            cohere_api_key="test-key",
            embedding_model="embed-multilingual-v3.0",
        )
        mock_client = MagicMock()
        mock_cohere_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.embeddings.float_ = [[0.1, 0.2]]

        import asyncio
        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.embed.return_value = future

        provider = CohereEmbeddingProvider()
        result = await provider.generate_embedding("test text")
        assert result == [0.1, 0.2]

    @pytest.mark.asyncio
    @patch("app.services.embedding_service.get_settings")
    @patch("cohere.AsyncClientV2")
    async def test_generate_embeddings_batch(self, mock_cohere_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            cohere_api_key="test-key",
            embedding_model="embed-multilingual-v3.0",
        )
        mock_client = MagicMock()
        mock_cohere_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.embeddings.float_ = [[0.1], [0.2]]

        import asyncio
        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.embed.return_value = future

        provider = CohereEmbeddingProvider()
        result = await provider.generate_embeddings_batch(["a", "b"])
        assert result == [[0.1], [0.2]]


class TestLocalEmbeddingProviderInit:
    @patch("app.services.embedding_service.get_settings")
    @patch("sentence_transformers.SentenceTransformer")
    def test_constructor_sets_fields(self, mock_st_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            embedding_model="intfloat/multilingual-e5-large",
        )
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st_cls.return_value = mock_model

        provider = LocalEmbeddingProvider()
        assert provider.dimensions == 1024

    @pytest.mark.asyncio
    @patch("app.services.embedding_service.get_settings")
    @patch("sentence_transformers.SentenceTransformer")
    async def test_generate_embedding(self, mock_st_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            embedding_model="intfloat/multilingual-e5-large",
        )
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2])
        mock_st_cls.return_value = mock_model

        provider = LocalEmbeddingProvider()
        result = await provider.generate_embedding("test text")
        assert result == [0.1, 0.2]

    @pytest.mark.asyncio
    @patch("app.services.embedding_service.get_settings")
    @patch("sentence_transformers.SentenceTransformer")
    async def test_generate_embeddings_batch(self, mock_st_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            embedding_model="intfloat/multilingual-e5-large",
        )
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1], [0.2]])
        mock_st_cls.return_value = mock_model

        provider = LocalEmbeddingProvider()
        result = await provider.generate_embeddings_batch(["a", "b"])
        assert result == [[0.1], [0.2]]
