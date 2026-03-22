import logging
import uuid

from fastapi import APIRouter, Depends

from app.auth.middleware import get_current_user
from app.agents.news_agent import run_agent
from app.models.chat import ChatRequest, ChatResponse, SourceReference, ToolCall

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
