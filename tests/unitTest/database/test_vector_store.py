import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.database.vector_store import (
    vector_search,
    text_search,
    reciprocal_rank_fusion,
    hybrid_search,
    ensure_indexes,
)

MOCK_SETTINGS = MagicMock(
    vector_search_index_name="news_vector_index",
    vector_search_num_candidates=100,
    vector_search_limit=10,
    text_search_index_name="news_text_index",
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


class TestTextSearch:
    @pytest.mark.asyncio
    @patch("app.database.vector_store.get_settings", return_value=MOCK_SETTINGS)
    @patch("app.database.vector_store.get_collection")
    async def test_returns_results(self, mock_get_coll, _mock_settings):
        mock_docs = [
            {"NewsId": "n1", "Title": "Article 1", "score": 8.5},
            {"NewsId": "n2", "Title": "Article 2", "score": 6.2},
        ]

        async def mock_aggregate(pipeline):
            for doc in mock_docs:
                yield doc

        mock_collection = MagicMock()
        mock_collection.aggregate = mock_aggregate
        mock_get_coll.return_value = mock_collection

        results = await text_search(query="politics")
        assert len(results) == 2
        assert results[0]["Title"] == "Article 1"

    @pytest.mark.asyncio
    @patch("app.database.vector_store.get_settings", return_value=MOCK_SETTINGS)
    @patch("app.database.vector_store.get_collection")
    async def test_empty_results(self, mock_get_coll, _mock_settings):
        async def mock_aggregate(pipeline):
            return
            yield

        mock_collection = MagicMock()
        mock_collection.aggregate = mock_aggregate
        mock_get_coll.return_value = mock_collection

        results = await text_search(query="nonexistent")
        assert results == []

    @pytest.mark.asyncio
    @patch("app.database.vector_store.get_settings", return_value=MOCK_SETTINGS)
    @patch("app.database.vector_store.get_collection")
    async def test_builds_correct_pipeline(self, mock_get_coll, _mock_settings):
        captured_pipeline = []

        async def mock_aggregate(pipeline):
            captured_pipeline.extend(pipeline)
            return
            yield

        mock_collection = MagicMock()
        mock_collection.aggregate = mock_aggregate
        mock_get_coll.return_value = mock_collection

        await text_search(query="test query", limit=5)
        assert "$search" in captured_pipeline[0]
        assert captured_pipeline[0]["$search"]["index"] == "news_text_index"
        assert captured_pipeline[0]["$search"]["text"]["query"] == "test query"
        assert captured_pipeline[2]["$limit"] == 5

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

        await text_search(query="test", pre_filter={"CategoryId": "sports"})
        match_stage = captured_pipeline[2]
        assert match_stage == {"$match": {"CategoryId": "sports"}}


class TestReciprocalRankFusion:
    def test_merges_two_lists(self):
        vector_results = [
            {"NewsId": "a", "Title": "A", "score": 0.95},
            {"NewsId": "b", "Title": "B", "score": 0.90},
            {"NewsId": "c", "Title": "C", "score": 0.85},
        ]
        text_results = [
            {"NewsId": "b", "Title": "B", "score": 12.5},
            {"NewsId": "d", "Title": "D", "score": 10.0},
            {"NewsId": "a", "Title": "A", "score": 8.0},
        ]

        merged = reciprocal_rank_fusion(vector_results, text_results, final_limit=10)

        # Both a and b appear in both lists, should be ranked highest
        merged_ids = [d["NewsId"] for d in merged]
        assert "a" in merged_ids
        assert "b" in merged_ids
        assert "c" in merged_ids
        assert "d" in merged_ids
        # a and b appear in both lists so should be top 2
        assert set(merged_ids[:2]) == {"a", "b"}

    def test_respects_final_limit(self):
        results = [{"NewsId": f"n{i}", "score": 1.0} for i in range(20)]
        merged = reciprocal_rank_fusion(results, final_limit=5)
        assert len(merged) == 5

    def test_empty_lists(self):
        merged = reciprocal_rank_fusion([], [])
        assert merged == []

    def test_single_list(self):
        results = [
            {"NewsId": "x", "Title": "X", "score": 0.9},
            {"NewsId": "y", "Title": "Y", "score": 0.8},
        ]
        merged = reciprocal_rank_fusion(results, final_limit=10)
        assert len(merged) == 2
        assert merged[0]["NewsId"] == "x"

    def test_adds_rrf_score_field(self):
        results = [{"NewsId": "a", "score": 0.9}]
        merged = reciprocal_rank_fusion(results, final_limit=10)
        assert "score" in merged[0]
        # RRF score = 1/(60+1) ≈ 0.016393
        assert abs(merged[0]["score"] - 1.0 / 61) < 0.001

    def test_skips_docs_without_news_id(self):
        results = [{"Title": "No ID", "score": 0.9}]
        merged = reciprocal_rank_fusion(results, final_limit=10)
        assert merged == []


class TestHybridSearch:
    @pytest.mark.asyncio
    @patch("app.database.vector_store.text_search", new_callable=AsyncMock)
    @patch("app.database.vector_store.vector_search", new_callable=AsyncMock)
    @patch("app.database.vector_store.get_settings", return_value=MOCK_SETTINGS)
    async def test_combines_vector_and_text(
        self, _mock_settings, mock_vector, mock_text
    ):
        mock_vector.return_value = [
            {"NewsId": "a", "Title": "A", "score": 0.95},
            {"NewsId": "b", "Title": "B", "score": 0.80},
        ]
        mock_text.return_value = [
            {"NewsId": "b", "Title": "B", "score": 12.0},
            {"NewsId": "c", "Title": "C", "score": 8.0},
        ]

        results = await hybrid_search(
            query="test query", query_embedding=[0.1] * 1536, limit=5
        )

        assert len(results) == 3
        # b is in both lists, should rank highest
        assert results[0]["NewsId"] == "b"
        mock_vector.assert_called_once()
        mock_text.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.database.vector_store.text_search", new_callable=AsyncMock)
    @patch("app.database.vector_store.vector_search", new_callable=AsyncMock)
    @patch("app.database.vector_store.get_settings", return_value=MOCK_SETTINGS)
    async def test_passes_pre_filter(
        self, _mock_settings, mock_vector, mock_text
    ):
        mock_vector.return_value = []
        mock_text.return_value = []

        pre_filter = {"CategoryId": "politics"}
        await hybrid_search(
            query="test",
            query_embedding=[0.1] * 10,
            pre_filter=pre_filter,
        )

        mock_vector.assert_called_once()
        assert mock_vector.call_args.kwargs["pre_filter"] == pre_filter
        mock_text.assert_called_once()
        assert mock_text.call_args.kwargs["pre_filter"] == pre_filter

    @pytest.mark.asyncio
    @patch("app.database.vector_store.text_search", new_callable=AsyncMock)
    @patch("app.database.vector_store.vector_search", new_callable=AsyncMock)
    @patch("app.database.vector_store.get_settings", return_value=MOCK_SETTINGS)
    async def test_returns_empty_when_no_results(
        self, _mock_settings, mock_vector, mock_text
    ):
        mock_vector.return_value = []
        mock_text.return_value = []

        results = await hybrid_search(
            query="nothing", query_embedding=[0.1] * 10
        )
        assert results == []
