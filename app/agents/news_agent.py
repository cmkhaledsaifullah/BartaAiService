"""
Agentic RAG pipeline for Barta AI.

This agent uses LangChain's tool-calling agent to decide which retrieval
strategy to use based on the user's query, then generates a grounded
response from the retrieved news articles.

The LLM used by the agent is determined by LLM_PROVIDER in .env.
"""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import get_settings
from app.constants import AGENT_SYSTEM_PROMPT, ERROR_UNKNOWN_LLM_PROVIDER, MSG_AGENT_PROCESSING_FAILED
from app.agents.tools import ALL_TOOLS

logger = logging.getLogger(__name__)


def _create_langchain_llm() -> BaseChatModel:
    """Create a LangChain chat model based on the configured LLM_PROVIDER."""
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.3,
            max_tokens=2048,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.3,
            max_tokens=2048,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=0.3,
            max_output_tokens=2048,
        )
    elif provider == "groq":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=2048,
        )
    elif provider == "ollama":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key="ollama",
            base_url=f"{settings.ollama_base_url}/v1",
            temperature=0.3,
            max_tokens=2048,
        )
    else:
        raise ValueError(
            ERROR_UNKNOWN_LLM_PROVIDER.format(
                provider=provider, supported="openai, anthropic, google, groq, ollama"
            )
        )


def create_news_agent() -> AgentExecutor:
    """Create and return the Agentic RAG agent executor."""
    settings = get_settings()

    llm = _create_langchain_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=settings.app_debug,
        max_iterations=5,  # Prevent infinite loops
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


async def run_agent(
    query: str,
    chat_history: list[dict] | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Run the agentic RAG pipeline for a user query.

    Args:
        query: The user's question or message.
        chat_history: Previous messages in the conversation.
        session_id: Optional session identifier for tracking.

    Returns:
        Dict with 'answer', 'sources', and 'intermediate_steps'.
    """
    agent_executor = create_news_agent()

    # Convert chat history to LangChain message format
    lc_history = []
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_history.append(AIMessage(content=content))

    logger.info("Running agent for query: %s (session: %s)", query[:100], session_id)

    result = await agent_executor.ainvoke(
        {"input": query, "chat_history": lc_history}
    )

    # Extract sources from intermediate steps
    sources = _extract_sources(result.get("intermediate_steps", []))

    return {
        "answer": result.get("output", MSG_AGENT_PROCESSING_FAILED),
        "sources": sources,
        "tool_calls": _summarize_steps(result.get("intermediate_steps", [])),
    }


def _extract_sources(steps: list) -> list[dict]:
    """Extract unique source articles referenced in the agent's tool calls."""
    sources = []
    seen_urls = set()
    for _action, observation in steps:
        # Try to parse the observation as JSON (our tool output format)
        try:
            import json
            articles = json.loads(observation)
            if isinstance(articles, list):
                for article in articles:
                    url = article.get("source_url", "")
                    if url and url != "N/A" and url not in seen_urls:
                        seen_urls.add(url)
                        sources.append(
                            {
                                "title": article.get("title", ""),
                                "url": url,
                                "published": article.get("published", ""),
                                "newspaper": article.get("newspaper", ""),
                            }
                        )
        except (json.JSONDecodeError, TypeError):
            continue
    return sources


def _summarize_steps(steps: list) -> list[dict]:
    """Summarize the agent's intermediate steps for transparency."""
    summary = []
    for action, _observation in steps:
        summary.append(
            {
                "tool": action.tool,
                "input": action.tool_input,
            }
        )
    return summary
