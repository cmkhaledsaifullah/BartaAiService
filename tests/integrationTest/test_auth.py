import pytest
from unittest.mock import patch, AsyncMock

from app.auth.token import create_access_token


@pytest.mark.asyncio
async def test_register_success(client):
    """Test successful user registration."""
    mock_collection = AsyncMock()
    mock_collection.find_one = AsyncMock(return_value=None)
    mock_collection.insert_one = AsyncMock()

    with patch("app.controllers.auth_controller.get_collection", return_value=mock_collection):
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecurePass1",
            },
        )
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test@example.com"


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    """Test registration with an already-used email."""
    mock_collection = AsyncMock()
    mock_collection.find_one = AsyncMock(return_value={"email": "test@example.com"})

    with patch("app.controllers.auth_controller.get_collection", return_value=mock_collection):
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecurePass1",
            },
        )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_register_weak_password(client):
    """Test registration with a weak password."""
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "weak",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_success(client):
    """Test successful login."""
    from app.auth.token import hash_password

    mock_collection = AsyncMock()
    mock_collection.find_one = AsyncMock(
        return_value={
            "_id": "user-123",
            "email": "test@example.com",
            "password_hash": hash_password("SecurePass1"),
            "is_active": True,
        }
    )

    with patch("app.controllers.auth_controller.get_collection", return_value=mock_collection):
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "SecurePass1"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    """Test login with wrong password."""
    from app.auth.token import hash_password

    mock_collection = AsyncMock()
    mock_collection.find_one = AsyncMock(
        return_value={
            "_id": "user-123",
            "email": "test@example.com",
            "password_hash": hash_password("SecurePass1"),
            "is_active": True,
        }
    )

    with patch("app.controllers.auth_controller.get_collection", return_value=mock_collection):
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "WrongPass1"},
        )
    assert response.status_code == 401
