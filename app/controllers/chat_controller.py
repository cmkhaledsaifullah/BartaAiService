import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.auth.middleware import get_current_user
from app.agents.news_agent import run_agent
from app.constants import ERROR_ARTICLE_NOT_FOUND, MSG_CLICK_LOGGED
from app.models.chat import (
    ChatRequest,
    ChatResponse,
    ClickLogRequest,
    ClickLogResponse,
    SourceReference,
    ToolCall,
)
from app.models.news import NewsArticleResponse
from app.services.news_service import (
    get_article_by_id,
    get_article_by_source_url,
    log_click,
)

logger = logging.getLogger(__name__)


class ChatController:
    """Controller for the AI chat endpoint (Agentic RAG pipeline)."""

    def __init__(self):
        self.router = APIRouter(prefix="/chat", tags=["chat"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "",
            self.chat,
            methods=["POST"],
            response_model=ChatResponse,
        )
        self.router.add_api_route(
            "/click-log",
            self.record_click,
            methods=["POST"],
            response_model=ClickLogResponse,
        )

    async def chat(
        self,
        payload: ChatRequest,
        current_user: dict = Depends(get_current_user),
    ):
        """
        Main chat endpoint — sends the user's message through the Agentic RAG pipeline.

        Requires a valid JWT token in the Authorization header.
        The agent will:
        1. Analyze the user's query
        2. Decide which retrieval tools to use
        3. Fetch relevant news articles from MongoDB
        4. Generate a grounded response with citations
        """
        session_id = payload.session_id or str(uuid.uuid4())

        # Convert conversation history to the format the agent expects
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in payload.conversation_history
        ]

        logger.info(
            "Chat request from user=%s session=%s query=%s",
            current_user.get("email", "unknown"),
            session_id,
            payload.message[:80],
        )

        result = await run_agent(
            query=payload.message,
            chat_history=history,
            session_id=session_id,
        )

        sources = [SourceReference(**s) for s in result.get("sources", [])]
        tool_calls = [ToolCall(**t) for t in result.get("tool_calls", [])]

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            tool_calls=tool_calls,
            session_id=session_id,
        )

    async def record_click(
        self,
        payload: ClickLogRequest,
        current_user: dict = Depends(get_current_user),
    ) -> ClickLogResponse:
        """Log a query-article click pair.

        - If **news_id** is provided: log the pair and return the full article.
        - If only **source_url** is provided: resolve the NewsId, log the pair,
          and return a confirmation (no article body — the user is navigating
          to the external source).
        """
        if payload.news_id:
            article = await get_article_by_id(payload.news_id)
            if not article:
                raise HTTPException(status_code=404, detail=ERROR_ARTICLE_NOT_FOUND)
            await log_click(
                query=payload.query,
                news_id=payload.news_id,
                source_url=article.get("SourceURL", ""),
            )
            logger.info(
                "Click-log (news_id): user=%s news_id=%s",
                current_user.get("email", "unknown"),
                payload.news_id,
            )
            return ClickLogResponse(
                message=MSG_CLICK_LOGGED,
                article=NewsArticleResponse(**article),
            )

        # source_url path
        article = await get_article_by_source_url(payload.source_url)
        news_id = article["NewsId"] if article else ""
        await log_click(
            query=payload.query,
            news_id=news_id,
            source_url=payload.source_url,
        )
        logger.info(
            "Click-log (source_url): user=%s url=%s",
            current_user.get("email", "unknown"),
            payload.source_url,
        )
        return ClickLogResponse(message=MSG_CLICK_LOGGED)
