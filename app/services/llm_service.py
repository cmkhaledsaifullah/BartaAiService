"""
Pluggable LLM service.

Supports multiple LLM providers configured via LLM_PROVIDER in .env:
  - openai     : OpenAI GPT-4o / GPT-4o-mini / GPT-4.1 family (API)
  - anthropic  : Anthropic Claude 3.5 Sonnet / Haiku (API)
  - google     : Google Gemini 2.0 Flash / 2.5 Pro (API)
  - groq       : Groq-hosted open models like Llama 3.3 (API, fast & cheap)
  - ollama     : Local models via Ollama (free, no API key)
"""

import logging
from abc import ABC, abstractmethod

from app.config import get_settings
from app.constants import ERROR_UNKNOWN_LLM_PROVIDER, LLM_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Base class for all LLM providers."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAILLMProvider(LLMProvider):
    """OpenAI API provider (GPT-4o, GPT-4o-mini, GPT-4.1 family)."""

    def __init__(self):
        from openai import AsyncOpenAI

        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model

    @property
    def model_name(self) -> str:
        return self._model

    async def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicLLMProvider(LLMProvider):
    """Anthropic API provider (Claude 3.5 Sonnet, Claude 3.5 Haiku)."""

    def __init__(self):
        from anthropic import AsyncAnthropic

        settings = get_settings()
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model

    @property
    def model_name(self) -> str:
        return self._model

    async def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        # Anthropic separates system from messages
        system_msg = ""
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                filtered_messages.append(msg)

        response = await self._client.messages.create(
            model=self._model,
            system=system_msg,
            messages=filtered_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# Google Gemini provider
# ---------------------------------------------------------------------------

class GoogleLLMProvider(LLMProvider):
    """Google Gemini API provider (Gemini 2.0 Flash, Gemini 2.5 Pro)."""

    def __init__(self):
        from google import genai

        settings = get_settings()
        self._client = genai.Client(api_key=settings.google_api_key)
        self._model = settings.llm_model

    @property
    def model_name(self) -> str:
        return self._model

    async def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        from google.genai import types

        # Convert messages to Gemini format
        system_instruction = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
            elif msg["role"] == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text


# ---------------------------------------------------------------------------
# Groq provider (OpenAI-compatible API for Llama, Mixtral, etc.)
# ---------------------------------------------------------------------------

class GroqLLMProvider(LLMProvider):
    """Groq API provider (Llama 3.3 70B, Mixtral, etc. — OpenAI-compatible)."""

    def __init__(self):
        from openai import AsyncOpenAI

        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = settings.llm_model

    @property
    def model_name(self) -> str:
        return self._model

    async def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Ollama provider (local, free — OpenAI-compatible API)
# ---------------------------------------------------------------------------

class OllamaLLMProvider(LLMProvider):
    """Ollama local provider (Llama 3, Mistral, Gemma, etc. — free, no API key)."""

    def __init__(self):
        from openai import AsyncOpenAI

        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key="ollama",
            base_url=f"{settings.ollama_base_url}/v1",
        )
        self._model = settings.llm_model

    @property
    def model_name(self) -> str:
        return self._model

    async def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Provider factory & public API
# ---------------------------------------------------------------------------

_provider: LLMProvider | None = None

LLM_PROVIDERS = {
    "openai": OpenAILLMProvider,
    "anthropic": AnthropicLLMProvider,
    "google": GoogleLLMProvider,
    "groq": GroqLLMProvider,
    "ollama": OllamaLLMProvider,
}


def get_llm_provider() -> LLMProvider:
    """Get or create the configured LLM provider (singleton)."""
    global _provider
    if _provider is None:
        settings = get_settings()
        provider_name = settings.llm_provider.lower()
        provider_class = LLM_PROVIDERS.get(provider_name)
        if provider_class is None:
            raise ValueError(
                ERROR_UNKNOWN_LLM_PROVIDER.format(
                    provider=provider_name,
                    supported=", ".join(LLM_PROVIDERS.keys()),
                )
            )
        _provider = provider_class()
        logger.info(
            "LLM provider initialized: %s (model: %s)",
            provider_name,
            _provider.model_name,
        )
    return _provider


async def chat_completion(
    messages: list[dict],
    context_articles: list[dict] | None = None,
    temperature: float = 0.3,
) -> str:
    """
    Generate a chat completion using the configured LLM provider.

    Args:
        messages: Conversation history (user + assistant messages).
        context_articles: Retrieved news articles to use as context.
        temperature: LLM temperature (lower = more factual).

    Returns:
        The assistant's response text.
    """
    provider = get_llm_provider()

    # Build the system message with context
    system_content = LLM_SYSTEM_PROMPT
    if context_articles:
        context_text = _format_articles_as_context(context_articles)
        system_content += f"\n\n--- RETRIEVED NEWS ARTICLES ---\n{context_text}\n--- END OF ARTICLES ---"

    full_messages = [{"role": "system", "content": system_content}] + messages

    return await provider.chat_completion(
        messages=full_messages,
        temperature=temperature,
    )


def _format_articles_as_context(articles: list[dict]) -> str:
    """Format retrieved articles into a text block for the LLM context window."""
    parts = []
    for i, article in enumerate(articles, 1):
        relevance = article.get("score", "N/A")
        relevance_str = f"{relevance:.4f}" if isinstance(relevance, (int, float)) else str(relevance)
        parts.append(
            f"[Article {i}] (Relevance: {relevance_str})\n"
            f"Title: {article.get('Title', 'N/A')}\n"
            f"Published: {article.get('PublishDate', 'N/A')}\n"
            f"Author: {article.get('Author', 'N/A')}\n"
            f"Source: {article.get('SourceURL', 'N/A')}\n"
            f"Category: {article.get('Category', 'N/A')}\n"
            f"Tags: {', '.join(article.get('Tags', []))}\n"
            f"Content:\n{article.get('Body', 'N/A')}\n"
        )
    return "\n---\n".join(parts)
