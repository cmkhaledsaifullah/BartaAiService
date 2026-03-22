import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.news_service import (
    get_article_by_id,
    search_articles_by_filter,
    get_recent_articles,
    get_articles_by_tags,
)


SAMPLE_ARTICLE = {
    "NewsId": "news-123",
    "Title": "Test Article",
    "CategoryId": "politics",
    "NewsPaperId": "daily_star",
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

        result = await search_articles_by_filter({"CategoryId": "politics"})
        assert len(result) == 1
        mock_collection.find.assert_called_once_with(
            {"CategoryId": "politics"}, {"_id": 0, "embedding": 0}
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
        mock_search.assert_called_once_with({"CategoryId": "sports"}, limit=10)

    @pytest.mark.asyncio
    @patch("app.services.news_service.search_articles_by_filter", new_callable=AsyncMock)
    async def test_with_newspaper_filter(self, mock_search):
        mock_search.return_value = []
        await get_recent_articles(limit=10, newspaper_id="daily_star")
        mock_search.assert_called_once_with({"NewsPaperId": "daily_star"}, limit=10)

    @pytest.mark.asyncio
    @patch("app.services.news_service.search_articles_by_filter", new_callable=AsyncMock)
    async def test_with_both_filters(self, mock_search):
        mock_search.return_value = []
        await get_recent_articles(limit=10, category_id="politics", newspaper_id="daily_star")
        mock_search.assert_called_once_with(
            {"CategoryId": "politics", "NewsPaperId": "daily_star"}, limit=10
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
