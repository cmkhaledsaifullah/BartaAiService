from pydantic import BaseModel, Field, model_validator

from app.constants import ERROR_CLICK_LOG_MISSING_IDENTIFIER
from app.models.news import NewsArticleResponse


class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The user's message/question",
    )
    conversation_history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=50,
        description="Previous messages in this conversation",
    )
    session_id: str | None = Field(
        None, description="Optional session ID for conversation tracking"
    )


class SourceReference(BaseModel):
    """A source article referenced in the response."""
    title: str
    url: str
    published: str = ""
    newspaper: str = ""


class ToolCall(BaseModel):
    """Record of a tool the agent invoked."""
    tool: str
    input: dict | str


class ChatResponse(BaseModel):
    """Response body from the chat endpoint."""
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    session_id: str | None = None


class ClickLogRequest(BaseModel):
    """Request body for logging a query-article click pair.

    The client must provide either news_id (user clicked the article directly)
    or source_url (user clicked the external source link). At least one is required.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The search query that produced this result",
    )
    news_id: str | None = Field(
        None, description="NewsId of the clicked article"
    )
    source_url: str | None = Field(
        None, description="Source URL the user clicked"
    )

    @model_validator(mode="after")
    def check_at_least_one_identifier(self):
        if not self.news_id and not self.source_url:
            raise ValueError(ERROR_CLICK_LOG_MISSING_IDENTIFIER)
        return self


class ClickLogResponse(BaseModel):
    """Response for a click-log request."""

    message: str
    article: NewsArticleResponse | None = None
