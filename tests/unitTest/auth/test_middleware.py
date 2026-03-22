import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from fastapi import HTTPException

from app.auth.middleware import get_current_user


MOCK_SETTINGS = MagicMock(
    jwt_secret_key="test-secret-key-for-unit-tests",
    jwt_algorithm="HS256",
    jwt_access_token_expire_minutes=60,
)


def _make_credentials(token: str):
    """Helper to create a mock HTTPAuthorizationCredentials."""
    creds = MagicMock()
    creds.credentials = token
    return creds


def _create_token(data: dict) -> str:
    """Create a valid JWT for testing."""
    with patch("app.auth.token.get_settings", return_value=MOCK_SETTINGS):
        from app.auth.token import create_access_token

        return create_access_token(data=data)


class TestGetCurrentUser:
    @pytest.mark.asyncio
    @patch("app.auth.middleware.get_collection")
    @patch("app.auth.middleware.decode_access_token")
    async def test_valid_token_returns_user(self, mock_decode, mock_get_coll):
        mock_decode.return_value = {"sub": "user-123", "email": "a@b.com"}
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(
            return_value={"_id": "user-123", "email": "a@b.com", "is_active": True}
        )
        mock_get_coll.return_value = mock_collection

        credentials = _make_credentials("valid-token")
        user = await get_current_user(credentials)

        assert user["_id"] == "user-123"
        mock_decode.assert_called_once_with("valid-token")

    @pytest.mark.asyncio
    @patch("app.auth.middleware.decode_access_token")
    async def test_token_without_sub_raises_401(self, mock_decode):
        mock_decode.return_value = {"email": "a@b.com"}  # no "sub" key

        credentials = _make_credentials("bad-token")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch("app.auth.middleware.decode_access_token")
    async def test_expired_token_raises_401(self, mock_decode):
        import jwt

        mock_decode.side_effect = jwt.ExpiredSignatureError()

        credentials = _make_credentials("expired-token")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch("app.auth.middleware.decode_access_token")
    async def test_invalid_token_raises_401(self, mock_decode):
        import jwt

        mock_decode.side_effect = jwt.InvalidTokenError()

        credentials = _make_credentials("garbage")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch("app.auth.middleware.get_collection")
    @patch("app.auth.middleware.decode_access_token")
    async def test_user_not_found_raises_401(self, mock_decode, mock_get_coll):
        mock_decode.return_value = {"sub": "missing-user"}
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_get_coll.return_value = mock_collection

        credentials = _make_credentials("valid-token")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch("app.auth.middleware.get_collection")
    @patch("app.auth.middleware.decode_access_token")
    async def test_deactivated_user_raises_403(self, mock_decode, mock_get_coll):
        mock_decode.return_value = {"sub": "user-123"}
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(
            return_value={"_id": "user-123", "is_active": False}
        )
        mock_get_coll.return_value = mock_collection

        credentials = _make_credentials("valid-token")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials)
        assert exc_info.value.status_code == 403
