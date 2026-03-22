import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock

from app.agents.news_agent import (
    _create_langchain_llm,
    _extract_sources,
    _summarize_steps,
    run_agent,
    create_news_agent,
)


MOCK_SETTINGS = MagicMock(
    llm_provider="openai",
    llm_model="gpt-4o",
    openai_api_key="fake-key",
    anthropic_api_key="fake-key",
    google_api_key="fake-key",
    groq_api_key="fake-key",
    app_debug=False,
)


class TestCreateLangchainLlm:
    @patch("app.agents.news_agent.get_settings", return_value=MagicMock(
        llm_provider="openai", llm_model="gpt-4o", openai_api_key="fake"))
    @patch("langchain_openai.ChatOpenAI")
    def test_openai_provider(self, mock_chat, _):
        _create_langchain_llm()
        mock_chat.assert_called_once()

    @patch("app.agents.news_agent.get_settings", return_value=MagicMock(
        llm_provider="anthropic", llm_model="claude-sonnet-4-20250514", anthropic_api_key="fake"))
    @patch("langchain_anthropic.ChatAnthropic")
    def test_anthropic_provider(self, mock_chat, _):
        _create_langchain_llm()
        mock_chat.assert_called_once()

    @patch("app.agents.news_agent.get_settings", return_value=MagicMock(
        llm_provider="google", llm_model="gemini-2.0-flash", google_api_key="fake"))
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_google_provider(self, mock_chat, _):
        _create_langchain_llm()
        mock_chat.assert_called_once()

    @patch("app.agents.news_agent.get_settings", return_value=MagicMock(
        llm_provider="groq", llm_model="llama-3.3-70b", groq_api_key="fake"))
    @patch("langchain_openai.ChatOpenAI")
    def test_groq_provider(self, mock_chat, _):
        _create_langchain_llm()
        mock_chat.assert_called_once()

    @patch("app.agents.news_agent.get_settings", return_value=MagicMock(llm_provider="unknown"))
    def test_unknown_provider_raises(self, _):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            _create_langchain_llm()


class TestExtractSources:
    def test_extracts_sources_from_json_observation(self):
        articles = [
            {"title": "A", "source_url": "https://example.com/a", "published": "2026-03-01", "newspaper": "daily_star"},
            {"title": "B", "source_url": "https://example.com/b", "published": "2026-03-02", "newspaper": "prothom_alo"},
        ]
        mock_action = MagicMock()
        steps = [(mock_action, json.dumps(articles))]

        result = _extract_sources(steps)
        assert len(result) == 2
        assert result[0]["title"] == "A"
        assert result[1]["url"] == "https://example.com/b"

    def test_deduplicates_by_url(self):
        articles = [
            {"title": "A", "source_url": "https://example.com/a"},
            {"title": "A dup", "source_url": "https://example.com/a"},
        ]
        mock_action = MagicMock()
        steps = [(mock_action, json.dumps(articles))]

        result = _extract_sources(steps)
        assert len(result) == 1

    def test_skips_na_urls(self):
        articles = [{"title": "No URL", "source_url": "N/A"}]
        mock_action = MagicMock()
        steps = [(mock_action, json.dumps(articles))]

        result = _extract_sources(steps)
        assert len(result) == 0

    def test_skips_empty_urls(self):
        articles = [{"title": "Empty URL", "source_url": ""}]
        mock_action = MagicMock()
        steps = [(mock_action, json.dumps(articles))]

        result = _extract_sources(steps)
        assert len(result) == 0

    def test_handles_non_json_observation_gracefully(self):
        mock_action = MagicMock()
        steps = [(mock_action, "plain text response")]

        result = _extract_sources(steps)
        assert result == []

    def test_handles_empty_steps(self):
        result = _extract_sources([])
        assert result == []


class TestSummarizeSteps:
    def test_summarizes_tool_calls(self):
        action1 = MagicMock(tool="semantic_news_search", tool_input={"query": "politics"})
        action2 = MagicMock(tool="get_latest_news", tool_input={"limit": 5})
        steps = [(action1, "obs1"), (action2, "obs2")]

        result = _summarize_steps(steps)
        assert len(result) == 2
        assert result[0]["tool"] == "semantic_news_search"
        assert result[1]["input"] == {"limit": 5}

    def test_empty_steps(self):
        result = _summarize_steps([])
        assert result == []


class TestRunAgent:
    @pytest.mark.asyncio
    @patch("app.agents.news_agent.create_news_agent")
    async def test_run_agent_returns_structured_result(self, mock_create):
        mock_executor = AsyncMock()
        mock_executor.ainvoke = AsyncMock(return_value={
            "output": "Here are the latest news...",
            "intermediate_steps": [],
        })
        mock_create.return_value = mock_executor

        result = await run_agent(query="latest news", chat_history=[], session_id="s1")
        assert result["answer"] == "Here are the latest news..."
        assert result["sources"] == []
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    @patch("app.agents.news_agent.create_news_agent")
    async def test_run_agent_with_chat_history(self, mock_create):
        mock_executor = AsyncMock()
        mock_executor.ainvoke = AsyncMock(return_value={
            "output": "Response",
            "intermediate_steps": [],
        })
        mock_create.return_value = mock_executor

        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = await run_agent(query="follow up", chat_history=history)
        assert result["answer"] == "Response"
        call_args = mock_executor.ainvoke.call_args[0][0]
        assert len(call_args["chat_history"]) == 2

    @pytest.mark.asyncio
    @patch("app.agents.news_agent.create_news_agent")
    async def test_run_agent_fallback_message(self, mock_create):
        mock_executor = AsyncMock()
        mock_executor.ainvoke = AsyncMock(return_value={
            "intermediate_steps": [],
        })
        mock_create.return_value = mock_executor

        result = await run_agent(query="test")
        assert result["answer"] == "I was unable to process your request."
