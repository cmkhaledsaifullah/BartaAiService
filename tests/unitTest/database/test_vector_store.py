import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.database.vector_store import vector_search, ensure_indexes

MOCK_SETTINGS = MagicMock(
    vector_search_index_name="news_vector_index",
    vector_search_num_candidates=100,
    vector_search_limit=10,
)


class TestVectorSearch:
    @pytest.mark.asyncio
    @patch("app.database.vector_store.get_settings", return_value=MOCK_SETTINGS)
    @patch("app.database.vector_store.get_collection")
    async def test_returns_results(self, mock_get_coll, _mock_settings):
        mock_docs = [
            {"Title": "Article 1", "score": 0.95},
            {"Title": "Article 2", "score": 0.90},
        ]

        async def mock_aggregate(pipeline):
            for doc in mock_docs:
                yield doc

        mock_collection = MagicMock()
        mock_collection.aggregate = mock_aggregate
        mock_get_coll.return_value = mock_collection

        results = await vector_search(query_embedding=[0.1] * 1536)
        assert len(results) == 2
        assert results[0]["Title"] == "Article 1"

    @pytest.mark.asyncio
    @patch("app.database.vector_store.get_settings", return_value=MOCK_SETTINGS)
    @patch("app.database.vector_store.get_collection")
    async def test_empty_results(self, mock_get_coll, _mock_settings):
        async def mock_aggregate(pipeline):
            return
            yield  # Make it an async generator

        mock_collection = MagicMock()
        mock_collection.aggregate = mock_aggregate
        mock_get_coll.return_value = mock_collection

        results = await vector_search(query_embedding=[0.1] * 1536)
        assert results == []

    @pytest.mark.asyncio
    @patch("app.database.vector_store.get_settings", return_value=MOCK_SETTINGS)
    @patch("app.database.vector_store.get_collection")
    async def test_custom_limit(self, mock_get_coll, _mock_settings):
        async def mock_aggregate(pipeline):
            assert pipeline[0]["$vectorSearch"]["limit"] == 5
            return
            yield

        mock_collection = MagicMock()
        mock_collection.aggregate = mock_aggregate
        mock_get_coll.return_value = mock_collection

        await vector_search(query_embedding=[0.1] * 1536, limit=5)

    @pytest.mark.asyncio
    @patch("app.database.vector_store.get_settings", return_value=MOCK_SETTINGS)
    @patch("app.database.vector_store.get_collection")
    async def test_with_pre_filter(self, mock_get_coll, _mock_settings):
        captured_pipeline = []

        async def mock_aggregate(pipeline):
            captured_pipeline.extend(pipeline)
            return
            yield

        mock_collection = MagicMock()
        mock_collection.aggregate = mock_aggregate
        mock_get_coll.return_value = mock_collection

        pre_filter = {"CategoryId": "politics"}
        await vector_search(query_embedding=[0.1] * 1536, pre_filter=pre_filter)
        assert captured_pipeline[0]["$vectorSearch"]["filter"] == {"CategoryId": "politics"}


class TestEnsureIndexes:
    @pytest.mark.asyncio
    @patch("app.database.vector_store.get_collection")
    async def test_creates_all_indexes(self, mock_get_coll):
        mock_collection = AsyncMock()
        mock_get_coll.return_value = mock_collection

        await ensure_indexes()

        assert mock_collection.create_index.call_count == 5
        index_calls = [c.args[0] for c in mock_collection.create_index.call_args_list]
        assert "NewsId" in index_calls
        assert "CategoryId" in index_calls
        assert "NewsPaperId" in index_calls
        assert "PublishDate" in index_calls
        assert "Tags" in index_calls
