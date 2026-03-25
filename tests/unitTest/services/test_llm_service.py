import pytest
from unittest.mock import patch, MagicMock, call

from app.services.llm_service import (
    get_llm_provider,
    chat_completion,
    _format_articles_as_context,
    LLM_PROVIDERS,
    OpenAILLMProvider,
    AnthropicLLMProvider,
    GoogleLLMProvider,
    GroqLLMProvider,
    OllamaLLMProvider,
)
import app.services.llm_service as llm_module


class TestLlmProviderFactory:
    def teardown_method(self):
        llm_module._provider = None

    @patch("app.services.llm_service.get_settings")
    def test_creates_openai_provider(self, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="openai", llm_model="gpt-4o")
        mock_instance = MagicMock(model_name="gpt-4o")
        with patch.dict(LLM_PROVIDERS, {"openai": lambda: mock_instance}):
            provider = get_llm_provider()
        assert provider is mock_instance

    @patch("app.services.llm_service.get_settings")
    def test_creates_anthropic_provider(self, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="anthropic", llm_model="claude-sonnet-4-20250514")
        mock_instance = MagicMock(model_name="claude-sonnet-4-20250514")
        with patch.dict(LLM_PROVIDERS, {"anthropic": lambda: mock_instance}):
            provider = get_llm_provider()
        assert provider is mock_instance

    @patch("app.services.llm_service.get_settings")
    def test_creates_google_provider(self, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="google", llm_model="gemini-2.0-flash")
        mock_instance = MagicMock(model_name="gemini-2.0-flash")
        with patch.dict(LLM_PROVIDERS, {"google": lambda: mock_instance}):
            provider = get_llm_provider()
        assert provider is mock_instance

    @patch("app.services.llm_service.get_settings")
    def test_creates_groq_provider(self, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="groq", llm_model="llama-3.3-70b")
        mock_instance = MagicMock(model_name="llama-3.3-70b")
        with patch.dict(LLM_PROVIDERS, {"groq": lambda: mock_instance}):
            provider = get_llm_provider()
        assert provider is mock_instance

    @patch("app.services.llm_service.get_settings")
    def test_creates_ollama_provider(self, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="ollama", llm_model="llama3.2")
        mock_instance = MagicMock(model_name="llama3.2")
        with patch.dict(LLM_PROVIDERS, {"ollama": lambda: mock_instance}):
            provider = get_llm_provider()
        assert provider is mock_instance

    @patch("app.services.llm_service.get_settings")
    def test_unknown_provider_raises(self, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="invalid")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider()

    @patch("app.services.llm_service.get_settings")
    def test_singleton(self, mock_settings):
        mock_settings.return_value = MagicMock(llm_provider="openai", llm_model="gpt-4o")
        mock_factory = MagicMock()
        mock_instance = MagicMock(model_name="gpt-4o")
        mock_factory.return_value = mock_instance
        with patch.dict(LLM_PROVIDERS, {"openai": mock_factory}):
            p1 = get_llm_provider()
            p2 = get_llm_provider()
        assert p1 is p2
        assert mock_factory.call_count == 1


class TestFormatArticlesAsContext:
    def test_formats_single_article(self):
        articles = [{
            "Title": "Test Title",
            "PublishDate": "2026-03-20",
            "Author": "Reporter",
            "SourceURL": "https://example.com",
            "Category": "politics",
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
        assert "ollama" in LLM_PROVIDERS
        assert len(LLM_PROVIDERS) == 5


class TestOpenAILLMProviderInit:
    @patch("app.services.llm_service.get_settings")
    @patch("openai.AsyncOpenAI")
    def test_constructor(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            openai_api_key="test-key", llm_model="gpt-4o"
        )
        provider = OpenAILLMProvider()
        assert provider.model_name == "gpt-4o"

    @pytest.mark.asyncio
    @patch("app.services.llm_service.get_settings")
    @patch("openai.AsyncOpenAI")
    async def test_chat_completion(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            openai_api_key="test-key", llm_model="gpt-4o"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]

        import asyncio
        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.chat.completions.create.return_value = future

        provider = OpenAILLMProvider()
        result = await provider.chat_completion(
            messages=[{"role": "user", "content": "hi"}]
        )
        assert result == "Hello!"


class TestAnthropicLLMProviderInit:
    @patch("app.services.llm_service.get_settings")
    @patch("anthropic.AsyncAnthropic")
    def test_constructor(self, mock_anthropic_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="test-key", llm_model="claude-sonnet-4-20250514"
        )
        provider = AnthropicLLMProvider()
        assert provider.model_name == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    @patch("app.services.llm_service.get_settings")
    @patch("anthropic.AsyncAnthropic")
    async def test_chat_completion(self, mock_anthropic_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="test-key", llm_model="claude-sonnet-4-20250514"
        )
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Anthropic response")]

        import asyncio
        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.messages.create.return_value = future

        provider = AnthropicLLMProvider()
        result = await provider.chat_completion(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        )
        assert result == "Anthropic response"


class TestGoogleLLMProviderInit:
    @patch("app.services.llm_service.get_settings")
    def test_constructor(self, mock_settings):
        mock_settings.return_value = MagicMock(
            google_api_key="test-key", llm_model="gemini-2.0-flash"
        )
        import sys
        mock_genai = MagicMock()
        sys.modules["google.genai"] = mock_genai
        try:
            provider = GoogleLLMProvider()
            assert provider.model_name == "gemini-2.0-flash"
        finally:
            del sys.modules["google.genai"]

    @pytest.mark.asyncio
    @patch("app.services.llm_service.get_settings")
    async def test_chat_completion(self, mock_settings):
        mock_settings.return_value = MagicMock(
            google_api_key="test-key", llm_model="gemini-2.0-flash"
        )
        import sys
        mock_genai = MagicMock()
        sys.modules["google.genai"] = mock_genai
        mock_types = MagicMock()
        sys.modules["google.genai.types"] = mock_types

        try:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            mock_response = MagicMock(text="Google response")
            import asyncio
            future = asyncio.Future()
            future.set_result(mock_response)
            mock_client.aio.models.generate_content.return_value = future

            provider = GoogleLLMProvider()
            provider._client = mock_client

            result = await provider.chat_completion(
                messages=[
                    {"role": "system", "content": "System msg"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "prev"},
                ]
            )
            assert result == "Google response"
        finally:
            del sys.modules["google.genai"]
            del sys.modules["google.genai.types"]


class TestGroqLLMProviderInit:
    @patch("app.services.llm_service.get_settings")
    @patch("openai.AsyncOpenAI")
    def test_constructor(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            groq_api_key="test-key", llm_model="llama-3.3-70b-versatile"
        )
        provider = GroqLLMProvider()
        assert provider.model_name == "llama-3.3-70b-versatile"

    @pytest.mark.asyncio
    @patch("app.services.llm_service.get_settings")
    @patch("openai.AsyncOpenAI")
    async def test_chat_completion(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            groq_api_key="test-key", llm_model="llama-3.3-70b-versatile"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Groq response"))]

        import asyncio
        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.chat.completions.create.return_value = future

        provider = GroqLLMProvider()
        result = await provider.chat_completion(
            messages=[{"role": "user", "content": "hi"}]
        )
        assert result == "Groq response"


class TestOllamaLLMProviderInit:
    @patch("app.services.llm_service.get_settings")
    @patch("openai.AsyncOpenAI")
    def test_constructor(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            ollama_base_url="http://localhost:11434", llm_model="llama3"
        )
        provider = OllamaLLMProvider()
        assert provider.model_name == "llama3"

    @pytest.mark.asyncio
    @patch("app.services.llm_service.get_settings")
    @patch("openai.AsyncOpenAI")
    async def test_chat_completion(self, mock_openai_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            ollama_base_url="http://localhost:11434", llm_model="llama3"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Ollama response"))]

        import asyncio
        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.chat.completions.create.return_value = future

        provider = OllamaLLMProvider()
        result = await provider.chat_completion(
            messages=[{"role": "user", "content": "hi"}]
        )
        assert result == "Ollama response"
