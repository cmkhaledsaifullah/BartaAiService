import pytest
from unittest.mock import patch, MagicMock

from app.services.llm_service import (
    get_llm_provider,
    chat_completion,
    _format_articles_as_context,
    LLM_PROVIDERS,
)
import app.services.llm_service as llm_module


class TestLlmProviderFactory:
    def teardown_method(self):
        llm_module._provider = None

    @patch("app.services.llm_service.get_settings")
    @patch("app.services.llm_service.OpenAILLMProvider")
    def test_creates_openai_provider(self, mock_cls, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="openai", llm_model="gpt-4o")
        mock_instance = MagicMock(model_name="gpt-4o")
        mock_cls.return_value = mock_instance

        provider = get_llm_provider()
        assert provider is mock_instance

    @patch("app.services.llm_service.get_settings")
    @patch("app.services.llm_service.AnthropicLLMProvider")
    def test_creates_anthropic_provider(self, mock_cls, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="anthropic", llm_model="claude-sonnet-4-20250514")
        mock_instance = MagicMock(model_name="claude-sonnet-4-20250514")
        mock_cls.return_value = mock_instance

        provider = get_llm_provider()
        assert provider is mock_instance

    @patch("app.services.llm_service.get_settings")
    @patch("app.services.llm_service.GoogleLLMProvider")
    def test_creates_google_provider(self, mock_cls, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="google", llm_model="gemini-2.0-flash")
        mock_instance = MagicMock(model_name="gemini-2.0-flash")
        mock_cls.return_value = mock_instance

        provider = get_llm_provider()
        assert provider is mock_instance

    @patch("app.services.llm_service.get_settings")
    @patch("app.services.llm_service.GroqLLMProvider")
    def test_creates_groq_provider(self, mock_cls, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="groq", llm_model="llama-3.3-70b")
        mock_instance = MagicMock(model_name="llama-3.3-70b")
        mock_cls.return_value = mock_instance

        provider = get_llm_provider()
        assert provider is mock_instance

    @patch("app.services.llm_service.get_settings")
    def test_unknown_provider_raises(self, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="invalid")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider()

    @patch("app.services.llm_service.get_settings")
    @patch("app.services.llm_service.OpenAILLMProvider")
    def test_singleton(self, mock_cls, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="openai", llm_model="gpt-4o")
        mock_instance = MagicMock(model_name="gpt-4o")
        mock_cls.return_value = mock_instance

        p1 = get_llm_provider()
        p2 = get_llm_provider()
        assert p1 is p2
        assert mock_cls.call_count == 1


class TestFormatArticlesAsContext:
    def test_formats_single_article(self):
        articles = [{
            "Title": "Test Title",
            "PublishDate": "2026-03-20",
            "Author": "Reporter",
            "SourceURL": "https://example.com",
            "CategoryId": "politics",
            "Tags": ["politics", "parliament"],
            "Body": "Article body.",
            "score": 0.95,
        }]
        result = _format_articles_as_context(articles)
        assert "Test Title" in result
        assert "0.9500" in result
        assert "politics, parliament" in result

    def test_formats_multiple_articles(self):
        articles = [
            {"Title": "A", "score": 0.9, "Tags": [], "Body": "Body A"},
            {"Title": "B", "score": 0.8, "Tags": [], "Body": "Body B"},
        ]
        result = _format_articles_as_context(articles)
        assert "[Article 1]" in result
        assert "[Article 2]" in result

    def test_handles_missing_fields(self):
        articles = [{}]
        result = _format_articles_as_context(articles)
        assert "N/A" in result

    def test_empty_list(self):
        result = _format_articles_as_context([])
        assert result == ""


class TestChatCompletion:
    def teardown_method(self):
        llm_module._provider = None

    @pytest.mark.asyncio
    async def test_chat_completion_without_context(self):
        import asyncio

        mock_provider = MagicMock()
        future = asyncio.Future()
        future.set_result("AI response")
        mock_provider.chat_completion.return_value = future
        llm_module._provider = mock_provider

        result = await chat_completion(
            messages=[{"role": "user", "content": "hello"}]
        )
        assert result == "AI response"
        call_kwargs = mock_provider.chat_completion.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_chat_completion_with_context_articles(self):
        import asyncio

        mock_provider = MagicMock()
        future = asyncio.Future()
        future.set_result("contextual response")
        mock_provider.chat_completion.return_value = future
        llm_module._provider = mock_provider

        articles = [{"Title": "News", "score": 0.9, "Tags": [], "Body": "Content"}]
        result = await chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            context_articles=articles,
        )
        assert result == "contextual response"
        system_msg = mock_provider.chat_completion.call_args[1]["messages"][0]["content"]
        assert "RETRIEVED NEWS ARTICLES" in system_msg


class TestLlmProvidersRegistry:
    def test_all_providers_registered(self):
        assert "openai" in LLM_PROVIDERS
        assert "anthropic" in LLM_PROVIDERS
        assert "google" in LLM_PROVIDERS
        assert "groq" in LLM_PROVIDERS
        assert len(LLM_PROVIDERS) == 4
