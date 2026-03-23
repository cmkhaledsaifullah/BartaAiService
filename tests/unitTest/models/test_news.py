import pytest
from pydantic import ValidationError

from app.models.news import StorageNewsArticle, NewsArticleResponse


class TestStorageNewsArticle:
    def test_valid_article(self):
        article = StorageNewsArticle(
            NewsId="n-123",
            NewsPaperId="daily_star",
            CategoryId="politics",
            Title="Test Title",
            Body="Test body",
            PublishDate="2026-03-20",
        )
        assert article.NewsId == "n-123"
        assert article.Tags == []
        assert article.Author == ""
        assert article.SourceURL == ""

    def test_with_all_fields(self):
        article = StorageNewsArticle(
            NewsId="n-123",
            NewsPaperId="daily_star",
            CategoryId="politics",
            Title="Test Title",
            Body="Test body",
            Tags=["politics", "parliament"],
            PublishDate="2026-03-20",
            Author="Reporter",
            SourceURL="https://example.com",
        )
        assert len(article.Tags) == 2
        assert article.Author == "Reporter"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            StorageNewsArticle(
                NewsPaperId="daily_star",
                CategoryId="politics",
                Title="No NewsId",
                Body="Body",
                PublishDate="2026-03-20",
            )


class TestNewsArticleResponse:
    def test_defaults(self):
        resp = NewsArticleResponse(
            NewsId="n-1",
            NewsPaperId="ds",
            CategoryId="politics",
            Title="Title",
            Body="Body",
            PublishDate="2026-03-20",
        )
        assert resp.score is None
        assert resp.Tags == []

    def test_with_score(self):
        resp = NewsArticleResponse(
            NewsId="n-1",
            NewsPaperId="ds",
            CategoryId="politics",
            Title="Title",
            Body="Body",
            PublishDate="2026-03-20",
            score=0.95,
        )
        assert resp.score == 0.95
