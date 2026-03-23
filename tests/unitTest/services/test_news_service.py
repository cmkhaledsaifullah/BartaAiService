import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.news_service import (
    get_article_by_id,
    get_article_by_source_url,
    search_articles_by_filter,
    get_recent_articles,
    get_articles_by_tags,
    log_click,
    ensure_click_log_indexes,
)


SAMPLE_ARTICLE = {
    "NewsId": "news-123",
    "Title": "Test Article",
    "Category": "politics",
    "NewsPaper": "daily_star",
    "Body": "Article body",
    "Tags": ["politics"],
    "PublishDate": "2026-03-20",
}


class TestGetArticleById:
    @pytest.mark.asyncio
    @patch("app.services.news_service.get_collection")
    async def test_returns_article(self, mock_get_coll):
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=SAMPLE_ARTICLE)
        mock_get_coll.return_value = mock_collection

        result = await get_article_by_id("news-123")
        assert result["NewsId"] == "news-123"
        mock_collection.find_one.assert_called_once_with(
            {"NewsId": "news-123"}, {"_id": 0, "embedding": 0}
        )

    @pytest.mark.asyncio
    @patch("app.services.news_service.get_collection")
    async def test_returns_none_when_not_found(self, mock_get_coll):
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_get_coll.return_value = mock_collection

        result = await get_article_by_id("missing")
        assert result is None


class TestSearchArticlesByFilter:
    @pytest.mark.asyncio
    @patch("app.services.news_service.get_collection")
    async def test_returns_articles(self, mock_get_coll):
        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.to_list = AsyncMock(return_value=[SAMPLE_ARTICLE])

        mock_collection = MagicMock()
        mock_collection.find = MagicMock(return_value=mock_cursor)
        mock_get_coll.return_value = mock_collection

        result = await search_articles_by_filter({"Category": "politics"})
        assert len(result) == 1
        mock_collection.find.assert_called_once_with(
            {"Category": "politics"}, {"_id": 0, "embedding": 0}
        )

    @pytest.mark.asyncio
    @patch("app.services.news_service.get_collection")
    async def test_custom_sort_and_limit(self, mock_get_coll):
        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.to_list = AsyncMock(return_value=[])

        mock_collection = MagicMock()
        mock_collection.find = MagicMock(return_value=mock_cursor)
        mock_get_coll.return_value = mock_collection

        await search_articles_by_filter({}, limit=5, sort_by="Title", sort_order=1)
        mock_cursor.sort.assert_called_once_with("Title", 1)
        mock_cursor.limit.assert_called_once_with(5)


class TestGetRecentArticles:
    @pytest.mark.asyncio
    @patch("app.services.news_service.search_articles_by_filter", new_callable=AsyncMock)
    async def test_no_filters(self, mock_search):
        mock_search.return_value = [SAMPLE_ARTICLE]
        result = await get_recent_articles(limit=5)
        assert len(result) == 1
        mock_search.assert_called_once_with({}, limit=5)

    @pytest.mark.asyncio
    @patch("app.services.news_service.search_articles_by_filter", new_callable=AsyncMock)
    async def test_with_category_filter(self, mock_search):
        mock_search.return_value = []
        await get_recent_articles(limit=10, category_id="sports")
        mock_search.assert_called_once_with({"Category": "sports"}, limit=10)

    @pytest.mark.asyncio
    @patch("app.services.news_service.search_articles_by_filter", new_callable=AsyncMock)
    async def test_with_newspaper_filter(self, mock_search):
        mock_search.return_value = []
        await get_recent_articles(limit=10, newspaper_id="daily_star")
        mock_search.assert_called_once_with({"NewsPaper": "daily_star"}, limit=10)

    @pytest.mark.asyncio
    @patch("app.services.news_service.search_articles_by_filter", new_callable=AsyncMock)
    async def test_with_both_filters(self, mock_search):
        mock_search.return_value = []
        await get_recent_articles(limit=10, category_id="politics", newspaper_id="daily_star")
        mock_search.assert_called_once_with(
            {"Category": "politics", "NewsPaper": "daily_star"}, limit=10
        )


class TestGetArticlesByTags:
    @pytest.mark.asyncio
    @patch("app.services.news_service.search_articles_by_filter", new_callable=AsyncMock)
    async def test_searches_with_in_operator(self, mock_search):
        mock_search.return_value = [SAMPLE_ARTICLE]
        result = await get_articles_by_tags(["politics", "parliament"], limit=10)
        assert len(result) == 1
        mock_search.assert_called_once_with(
            {"Tags": {"$in": ["politics", "parliament"]}}, limit=10
        )


class TestGetArticleBySourceUrl:
    @pytest.mark.asyncio
    @patch("app.services.news_service.get_collection")
    async def test_returns_article(self, mock_get_coll):
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(
            return_value={"NewsId": "n1", "Title": "Test"}
        )
        mock_get_coll.return_value = mock_collection

        result = await get_article_by_source_url("https://example.com/a")
        assert result["NewsId"] == "n1"
        mock_collection.find_one.assert_called_once_with(
            {"SourceURL": "https://example.com/a"}, {"_id": 0, "embedding": 0}
        )

    @pytest.mark.asyncio
    @patch("app.services.news_service.get_collection")
    async def test_returns_none_when_not_found(self, mock_get_coll):
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_get_coll.return_value = mock_collection

        result = await get_article_by_source_url("https://missing.com")
        assert result is None


class TestLogClick:
    @pytest.mark.asyncio
    @patch("app.services.news_service.get_collection")
    async def test_inserts_click_record(self, mock_get_coll):
        mock_collection = AsyncMock()
        mock_get_coll.return_value = mock_collection

        await log_click(
            query="latest news", news_id="news-123", source_url="https://example.com"
        )

        mock_get_coll.assert_called_with("click_logs")
        mock_collection.insert_one.assert_called_once()

        inserted = mock_collection.insert_one.call_args[0][0]
        assert inserted["query"] == "latest news"
        assert inserted["news_id"] == "news-123"
        assert inserted["source_url"] == "https://example.com"
        assert "clicked_at" in inserted

    @pytest.mark.asyncio
    @patch("app.services.news_service.get_collection")
    async def test_inserts_with_default_source_url(self, mock_get_coll):
        mock_collection = AsyncMock()
        mock_get_coll.return_value = mock_collection

        await log_click(query="test", news_id="n1")

        inserted = mock_collection.insert_one.call_args[0][0]
        assert inserted["source_url"] == ""


class TestEnsureClickLogIndexes:
    @pytest.mark.asyncio
    @patch("app.services.news_service.get_collection")
    async def test_creates_indexes(self, mock_get_coll):
        mock_collection = AsyncMock()
        mock_get_coll.return_value = mock_collection

        await ensure_click_log_indexes()

        assert mock_collection.create_index.call_count == 2
        index_calls = [c.args[0] for c in mock_collection.create_index.call_args_list]
        assert "news_id" in index_calls
        assert "clicked_at" in index_calls
