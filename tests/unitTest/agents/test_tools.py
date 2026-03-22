import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock

from app.agents.tools import (
    semantic_news_search,
    search_news_by_category,
    search_news_by_date_range,
    search_news_by_tags,
    search_news_by_newspaper,
    get_latest_news,
    _format_results,
)
from app.constants import (
    MSG_NO_ARTICLES_FOUND,
    MSG_NO_ARTICLES_BY_CATEGORY,
    MSG_NO_ARTICLES_BY_DATE_RANGE,
    MSG_NO_ARTICLES_BY_TAGS,
    MSG_NO_ARTICLES_BY_NEWSPAPER,
    MSG_NO_RECENT_ARTICLES,
    MSG_NO_TAGS_PROVIDED,
    MSG_INVALID_DATE_FORMAT,
)


SAMPLE_ARTICLE = {
    "Title": "Test Article",
    "PublishDate": "2026-03-20",
    "Author": "Reporter",
    "SourceURL": "https://example.com/article",
    "CategoryId": "politics",
    "NewsPaperId": "daily_star",
    "Tags": ["politics"],
    "Body": "Article body text.",
}


class TestFormatResults:
    def test_formats_single_article(self):
        result = _format_results([SAMPLE_ARTICLE])
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["title"] == "Test Article"
        assert parsed[0]["index"] == 1

    def test_formats_multiple_articles(self):
        result = _format_results([SAMPLE_ARTICLE, SAMPLE_ARTICLE])
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[1]["index"] == 2

    def test_includes_score_when_present(self):
        article_with_score = {**SAMPLE_ARTICLE, "score": 0.9512}
        result = _format_results([article_with_score])
        parsed = json.loads(result)
        assert parsed[0]["relevance_score"] == 0.9512

    def test_truncates_long_body(self):
        long_article = {**SAMPLE_ARTICLE, "Body": "x" * 5000}
        result = _format_results([long_article])
        parsed = json.loads(result)
        assert len(parsed[0]["body"]) == 2000

    def test_empty_list(self):
        result = _format_results([])
        parsed = json.loads(result)
        assert parsed == []


class TestSemanticNewsSearch:
    @pytest.mark.asyncio
    @patch("app.agents.tools.vector_search", new_callable=AsyncMock, return_value=[SAMPLE_ARTICLE])
    @patch("app.agents.tools.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 1536)
    async def test_returns_results(self, _mock_embed, _mock_search):
        result = await semantic_news_search.ainvoke({"query": "politics"})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["title"] == "Test Article"

    @pytest.mark.asyncio
    @patch("app.agents.tools.vector_search", new_callable=AsyncMock, return_value=[])
    @patch("app.agents.tools.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 1536)
    async def test_no_results(self, _mock_embed, _mock_search):
        result = await semantic_news_search.ainvoke({"query": "nothing"})
        assert result == MSG_NO_ARTICLES_FOUND


class TestSearchNewsByCategory:
    @pytest.mark.asyncio
    @patch("app.agents.tools.get_recent_articles", new_callable=AsyncMock, return_value=[SAMPLE_ARTICLE])
    async def test_returns_results(self, _mock):
        result = await search_news_by_category.ainvoke({"category_id": "politics"})
        parsed = json.loads(result)
        assert len(parsed) == 1

    @pytest.mark.asyncio
    @patch("app.agents.tools.get_recent_articles", new_callable=AsyncMock, return_value=[])
    async def test_no_results(self, _mock):
        result = await search_news_by_category.ainvoke({"category_id": "sports"})
        assert "sports" in result

    @pytest.mark.asyncio
    @patch("app.agents.tools.get_recent_articles", new_callable=AsyncMock, return_value=[SAMPLE_ARTICLE])
    async def test_caps_limit_at_20(self, mock_recent):
        await search_news_by_category.ainvoke({"category_id": "politics", "limit": 50})
        mock_recent.assert_called_once_with(limit=20, category_id="politics")


class TestSearchNewsByDateRange:
    @pytest.mark.asyncio
    @patch("app.agents.tools.search_articles_by_filter", new_callable=AsyncMock, return_value=[SAMPLE_ARTICLE])
    async def test_valid_dates(self, _mock):
        result = await search_news_by_date_range.ainvoke({
            "start_date": "2026-03-01", "end_date": "2026-03-20"
        })
        parsed = json.loads(result)
        assert len(parsed) == 1

    @pytest.mark.asyncio
    async def test_invalid_date_format(self):
        result = await search_news_by_date_range.ainvoke({
            "start_date": "invalid", "end_date": "2026-03-20"
        })
        assert result == MSG_INVALID_DATE_FORMAT

    @pytest.mark.asyncio
    @patch("app.agents.tools.search_articles_by_filter", new_callable=AsyncMock, return_value=[])
    async def test_no_results(self, _mock):
        result = await search_news_by_date_range.ainvoke({
            "start_date": "2020-01-01", "end_date": "2020-01-02"
        })
        assert "2020-01-01" in result


class TestSearchNewsByTags:
    @pytest.mark.asyncio
    @patch("app.agents.tools.get_articles_by_tags", new_callable=AsyncMock, return_value=[SAMPLE_ARTICLE])
    async def test_returns_results(self, _mock):
        result = await search_news_by_tags.ainvoke({"tags": "politics,parliament"})
        parsed = json.loads(result)
        assert len(parsed) == 1

    @pytest.mark.asyncio
    async def test_empty_tags(self):
        result = await search_news_by_tags.ainvoke({"tags": "  ,  , "})
        assert result == MSG_NO_TAGS_PROVIDED

    @pytest.mark.asyncio
    @patch("app.agents.tools.get_articles_by_tags", new_callable=AsyncMock, return_value=[])
    async def test_no_results(self, _mock):
        result = await search_news_by_tags.ainvoke({"tags": "nonexistent"})
        assert "nonexistent" in result


class TestSearchNewsByNewspaper:
    @pytest.mark.asyncio
    @patch("app.agents.tools.get_recent_articles", new_callable=AsyncMock, return_value=[SAMPLE_ARTICLE])
    async def test_returns_results(self, _mock):
        result = await search_news_by_newspaper.ainvoke({"newspaper_id": "daily_star"})
        parsed = json.loads(result)
        assert len(parsed) == 1

    @pytest.mark.asyncio
    @patch("app.agents.tools.get_recent_articles", new_callable=AsyncMock, return_value=[])
    async def test_no_results(self, _mock):
        result = await search_news_by_newspaper.ainvoke({"newspaper_id": "unknown"})
        assert "unknown" in result

    @pytest.mark.asyncio
    @patch("app.agents.tools.get_recent_articles", new_callable=AsyncMock, return_value=[SAMPLE_ARTICLE])
    async def test_caps_limit_at_20(self, mock_recent):
        await search_news_by_newspaper.ainvoke({"newspaper_id": "daily_star", "limit": 100})
        mock_recent.assert_called_once_with(limit=20, newspaper_id="daily_star")


class TestGetLatestNews:
    @pytest.mark.asyncio
    @patch("app.agents.tools.get_recent_articles", new_callable=AsyncMock, return_value=[SAMPLE_ARTICLE])
    async def test_returns_results(self, _mock):
        result = await get_latest_news.ainvoke({})
        parsed = json.loads(result)
        assert len(parsed) == 1

    @pytest.mark.asyncio
    @patch("app.agents.tools.get_recent_articles", new_callable=AsyncMock, return_value=[])
    async def test_no_results(self, _mock):
        result = await get_latest_news.ainvoke({})
        assert result == MSG_NO_RECENT_ARTICLES

    @pytest.mark.asyncio
    @patch("app.agents.tools.get_recent_articles", new_callable=AsyncMock, return_value=[SAMPLE_ARTICLE])
    async def test_caps_limit_at_15(self, mock_recent):
        await get_latest_news.ainvoke({"limit": 50})
        mock_recent.assert_called_once_with(limit=15)
