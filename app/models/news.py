from pydantic import BaseModel, Field
from datetime import datetime


class StorageNewsArticle(BaseModel):
    """Schema for a news article stored in MongoDB."""
    NewsId: str
    NewsPaperId: str
    CategoryId: str
    Title: str
    Body: str
    Tags: list[str] = Field(default_factory=list)
    PublishDate: str
    Author: str = ""
    SourceURL: str = ""


class NewsArticleResponse(BaseModel):
    """Schema for a news article in API responses (no embedding)."""
    NewsId: str
    NewsPaperId: str
    CategoryId: str
    Title: str
    Body: str
    Tags: list[str] = Field(default_factory=list)
    PublishDate: str
    Author: str = ""
    SourceURL: str = ""
    score: float | None = None
