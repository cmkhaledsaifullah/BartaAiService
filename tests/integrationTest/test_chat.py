import pytest
from unittest.mock import patch, AsyncMock

from app.auth.token import create_access_token


@pytest.mark.asyncio
async def test_chat_requires_auth(client):
    """Test that chat endpoint requires authentication."""
    response = await client.post(
        "/api/v1/chat",
        json={"message": "What is the latest news?"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_chat_with_invalid_token(client):
    """Test chat endpoint rejects invalid tokens."""
    response = await client.post(
        "/api/v1/chat",
        json={"message": "What is the latest news?"},
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_success(client):
    """Test successful chat with valid token and mocked agent."""
    token = create_access_token(data={"sub": "user-123", "email": "test@example.com"})

    mock_user = {
        "_id": "user-123",
        "username": "testuser",
        "email": "test@example.com",
        "is_active": True,
    }

    mock_agent_result = {
        "answer": "Here are the latest news articles about Bangladesh...",
        "sources": [
            {
                "title": "Test Article",
                "url": "https://example.com/article",
                "published": "2026-03-20",
                "newspaper": "Daily Star",
            }
        ],
        "tool_calls": [
            {"tool": "semantic_news_search", "input": {"query": "latest news"}}
        ],
    }

    with patch(
        "app.auth.middleware.get_collection"
    ) as mock_get_collection:
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=mock_user)
        mock_get_collection.return_value = mock_collection

        with patch(
            "app.controllers.chat_controller.run_agent",
            new_callable=AsyncMock,
            return_value=mock_agent_result,
        ):
            response = await client.post(
                "/api/v1/chat",
                json={"message": "What is the latest news?"},
                headers={"Authorization": f"Bearer {token}"},
            )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["title"] == "Test Article"


@pytest.mark.asyncio
async def test_chat_empty_message(client):
    """Test chat rejects empty messages."""
    token = create_access_token(data={"sub": "user-123", "email": "test@example.com"})
    response = await client.post(
        "/api/v1/chat",
        json={"message": ""},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422


# --- Click-Log Integration Tests ---


@pytest.mark.asyncio
async def test_click_log_requires_auth(client):
    """Test that click-log endpoint requires authentication."""
    response = await client.post(
        "/api/v1/chat/click-log",
        json={"query": "politics", "news_id": "n1"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_click_log_with_invalid_token(client):
    """Test click-log endpoint rejects invalid tokens."""
    response = await client.post(
        "/api/v1/chat/click-log",
        json={"query": "politics", "news_id": "n1"},
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_click_log_by_news_id_success(client):
    """Test successful click-log with news_id returns article."""
    token = create_access_token(data={"sub": "user-123", "email": "test@example.com"})

    mock_user = {
        "_id": "user-123",
        "username": "testuser",
        "email": "test@example.com",
        "is_active": True,
    }
    mock_article = {
        "NewsId": "news-123",
        "NewsPaperId": "daily_star",
        "CategoryId": "politics",
        "Title": "Test Article",
        "Body": "Article body",
        "Tags": ["politics"],
        "PublishDate": "2026-03-20",
        "Author": "Reporter",
        "SourceURL": "https://example.com/article",
    }

    with patch("app.auth.middleware.get_collection") as mock_auth_coll:
        mock_auth_collection = AsyncMock()
        mock_auth_collection.find_one = AsyncMock(return_value=mock_user)
        mock_auth_coll.return_value = mock_auth_collection

        with patch(
            "app.controllers.chat_controller.get_article_by_id",
            new_callable=AsyncMock,
            return_value=mock_article,
        ):
            with patch(
                "app.controllers.chat_controller.log_click",
                new_callable=AsyncMock,
            ) as mock_log:
                response = await client.post(
                    "/api/v1/chat/click-log",
                    json={"query": "politics news", "news_id": "news-123"},
                    headers={"Authorization": f"Bearer {token}"},
                )

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Click logged successfully."
    assert data["article"]["NewsId"] == "news-123"
    mock_log.assert_called_once()


@pytest.mark.asyncio
async def test_click_log_by_news_id_not_found(client):
    """Test click-log returns 404 when article not found."""
    token = create_access_token(data={"sub": "user-123", "email": "test@example.com"})

    mock_user = {
        "_id": "user-123",
        "username": "testuser",
        "email": "test@example.com",
        "is_active": True,
    }

    with patch("app.auth.middleware.get_collection") as mock_auth_coll:
        mock_auth_collection = AsyncMock()
        mock_auth_collection.find_one = AsyncMock(return_value=mock_user)
        mock_auth_coll.return_value = mock_auth_collection

        with patch(
            "app.controllers.chat_controller.get_article_by_id",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.post(
                "/api/v1/chat/click-log",
                json={"query": "test", "news_id": "missing"},
                headers={"Authorization": f"Bearer {token}"},
            )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_click_log_by_source_url_success(client):
    """Test click-log with source_url logs and returns confirmation."""
    token = create_access_token(data={"sub": "user-123", "email": "test@example.com"})

    mock_user = {
        "_id": "user-123",
        "username": "testuser",
        "email": "test@example.com",
        "is_active": True,
    }
    mock_article = {"NewsId": "news-123"}

    with patch("app.auth.middleware.get_collection") as mock_auth_coll:
        mock_auth_collection = AsyncMock()
        mock_auth_collection.find_one = AsyncMock(return_value=mock_user)
        mock_auth_coll.return_value = mock_auth_collection

        with patch(
            "app.controllers.chat_controller.get_article_by_source_url",
            new_callable=AsyncMock,
            return_value=mock_article,
        ):
            with patch(
                "app.controllers.chat_controller.log_click",
                new_callable=AsyncMock,
            ):
                response = await client.post(
                    "/api/v1/chat/click-log",
                    json={
                        "query": "test",
                        "source_url": "https://example.com/article",
                    },
                    headers={"Authorization": f"Bearer {token}"},
                )

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Click logged successfully."
    assert data["article"] is None


@pytest.mark.asyncio
async def test_click_log_missing_identifiers(client):
    """Test click-log rejects requests missing both news_id and source_url."""
    token = create_access_token(data={"sub": "user-123", "email": "test@example.com"})
    response = await client.post(
        "/api/v1/chat/click-log",
        json={"query": "test"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422
