import pytest
from pydantic import ValidationError

from app.models.chat import ChatMessage, ChatRequest, ChatResponse, SourceReference, ToolCall


class TestChatMessage:
    def test_valid_user_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"

    def test_valid_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.role == "assistant"

    def test_invalid_role(self):
        with pytest.raises(ValidationError, match="role"):
            ChatMessage(role="system", content="Not allowed")

    def test_empty_content(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")


class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(message="What is the latest news?")
        assert req.message == "What is the latest news?"
        assert req.conversation_history == []
        assert req.session_id is None

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_with_session_id(self):
        req = ChatRequest(message="hello", session_id="s-123")
        assert req.session_id == "s-123"

    def test_with_conversation_history(self):
        req = ChatRequest(
            message="follow up",
            conversation_history=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "response"},
            ],
        )
        assert len(req.conversation_history) == 2


class TestSourceReference:
    def test_minimal_source(self):
        ref = SourceReference(title="Article", url="https://example.com")
        assert ref.published == ""
        assert ref.newspaper == ""

    def test_full_source(self):
        ref = SourceReference(
            title="A", url="https://x.com", published="2026-03-20", newspaper="daily_star"
        )
        assert ref.newspaper == "daily_star"


class TestToolCall:
    def test_with_dict_input(self):
        tc = ToolCall(tool="semantic_news_search", input={"query": "politics"})
        assert tc.tool == "semantic_news_search"
        assert tc.input == {"query": "politics"}

    def test_with_string_input(self):
        tc = ToolCall(tool="get_latest_news", input="5")
        assert tc.input == "5"


class TestChatResponse:
    def test_minimal_response(self):
        resp = ChatResponse(answer="Here are the news...")
        assert resp.sources == []
        assert resp.tool_calls == []
        assert resp.session_id is None

    def test_full_response(self):
        resp = ChatResponse(
            answer="Response",
            sources=[SourceReference(title="A", url="https://x.com")],
            tool_calls=[ToolCall(tool="search", input={})],
            session_id="s1",
        )
        assert len(resp.sources) == 1
        assert resp.session_id == "s1"
