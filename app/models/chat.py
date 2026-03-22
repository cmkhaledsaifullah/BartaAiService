from pydantic import BaseModel, Field


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
