import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.controllers.chat_controller import ChatController
from app.models.chat import ClickLogRequest


class TestChatControllerChat:
    def setup_method(self):
        self.controller = ChatController()

    @pytest.mark.asyncio
    @patch("app.controllers.chat_controller.run_agent", new_callable=AsyncMock)
    async def test_chat_success(self, mock_run_agent):
        mock_run_agent.return_value = {
            "answer": "Here are the latest news...",
            "sources": [
                {"title": "Article 1", "url": "https://example.com/1", "published": "2026-03-20", "newspaper": "daily_star"}
            ],
            "tool_calls": [
                {"tool": "semantic_news_search", "input": {"query": "news"}}
            ],
        }

        payload = MagicMock(
            message="What is the latest news?",
            conversation_history=[],
            session_id="session-123",
        )
        current_user = {"email": "user@example.com"}

        result = await self.controller.chat(payload, current_user)

        assert result.answer == "Here are the latest news..."
        assert len(result.sources) == 1
        assert result.session_id == "session-123"

    @pytest.mark.asyncio
    @patch("app.controllers.chat_controller.run_agent", new_callable=AsyncMock)
    async def test_chat_generates_session_id_when_none(self, mock_run_agent):
        mock_run_agent.return_value = {
            "answer": "response",
            "sources": [],
            "tool_calls": [],
        }

        payload = MagicMock(
            message="hello",
            conversation_history=[],
            session_id=None,
        )
        current_user = {"email": "user@example.com"}

        result = await self.controller.chat(payload, current_user)

        assert result.session_id is not None
        assert len(result.session_id) > 0

    @pytest.mark.asyncio
    @patch("app.controllers.chat_controller.run_agent", new_callable=AsyncMock)
    async def test_chat_passes_conversation_history(self, mock_run_agent):
        mock_run_agent.return_value = {
            "answer": "response",
            "sources": [],
            "tool_calls": [],
        }

        msg1 = MagicMock(role="user", content="first question")
        msg2 = MagicMock(role="assistant", content="first answer")
        payload = MagicMock(
            message="follow up",
            conversation_history=[msg1, msg2],
            session_id="s1",
        )
        current_user = {"email": "user@example.com"}

        await self.controller.chat(payload, current_user)

        call_kwargs = mock_run_agent.call_args[1]
        assert len(call_kwargs["chat_history"]) == 2
        assert call_kwargs["chat_history"][0]["role"] == "user"


SAMPLE_ARTICLE = {
    "NewsId": "news-123",
    "NewsPaperId": "daily_star",
    "CategoryId": "politics",
    "Title": "Test Article",
    "Body": "Article body text",
    "Tags": ["politics"],
    "PublishDate": "2026-03-20",
    "Author": "Reporter",
    "SourceURL": "https://example.com/article",
}

MOCK_USER = {"email": "test@example.com"}


class TestRecordClick:
    def setup_method(self):
        self.controller = ChatController()

    @pytest.mark.asyncio
    @patch("app.controllers.chat_controller.log_click", new_callable=AsyncMock)
    @patch(
        "app.controllers.chat_controller.get_article_by_id",
        new_callable=AsyncMock,
    )
    async def test_click_by_news_id_returns_article(
        self, mock_get_article, mock_log_click
    ):
        mock_get_article.return_value = SAMPLE_ARTICLE

        payload = ClickLogRequest(query="politics news", news_id="news-123")
        result = await self.controller.record_click(payload, current_user=MOCK_USER)

        assert result.message == "Click logged successfully."
        assert result.article is not None
        assert result.article.NewsId == "news-123"
        mock_log_click.assert_called_once_with(
            query="politics news",
            news_id="news-123",
            source_url="https://example.com/article",
        )

    @pytest.mark.asyncio
    @patch("app.controllers.chat_controller.log_click", new_callable=AsyncMock)
    @patch(
        "app.controllers.chat_controller.get_article_by_id",
        new_callable=AsyncMock,
    )
    async def test_click_by_news_id_not_found_raises_404(
        self, mock_get_article, mock_log_click
    ):
        mock_get_article.return_value = None

        payload = ClickLogRequest(query="test", news_id="missing-id")
        with pytest.raises(Exception) as exc_info:
            await self.controller.record_click(payload, current_user=MOCK_USER)
        assert exc_info.value.status_code == 404
        mock_log_click.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.controllers.chat_controller.log_click", new_callable=AsyncMock)
    @patch(
        "app.controllers.chat_controller.get_article_by_source_url",
        new_callable=AsyncMock,
    )
    async def test_click_by_source_url_logs_with_resolved_news_id(
        self, mock_get_by_url, mock_log_click
    ):
        mock_get_by_url.return_value = SAMPLE_ARTICLE

        payload = ClickLogRequest(
            query="test query", source_url="https://example.com/article"
        )
        result = await self.controller.record_click(payload, current_user=MOCK_USER)

        assert result.message == "Click logged successfully."
        assert result.article is None
        mock_log_click.assert_called_once_with(
            query="test query",
            news_id="news-123",
            source_url="https://example.com/article",
        )

    @pytest.mark.asyncio
    @patch("app.controllers.chat_controller.log_click", new_callable=AsyncMock)
    @patch(
        "app.controllers.chat_controller.get_article_by_source_url",
        new_callable=AsyncMock,
    )
    async def test_click_by_source_url_unresolved_logs_empty_news_id(
        self, mock_get_by_url, mock_log_click
    ):
        mock_get_by_url.return_value = None

        payload = ClickLogRequest(
            query="test", source_url="https://unknown.com/article"
        )
        result = await self.controller.record_click(payload, current_user=MOCK_USER)

        assert result.message == "Click logged successfully."
        mock_log_click.assert_called_once_with(
            query="test",
            news_id="",
            source_url="https://unknown.com/article",
        )
