import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.controllers.chat_controller import ChatController


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
