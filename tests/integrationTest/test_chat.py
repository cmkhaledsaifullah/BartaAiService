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
