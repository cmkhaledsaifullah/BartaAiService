import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import HTTPException

from app.controllers.auth_controller import AuthController

MOCK_SETTINGS = MagicMock(jwt_access_token_expire_minutes=60)


class TestAuthControllerRegister:
    def setup_method(self):
        self.controller = AuthController()

    @pytest.mark.asyncio
    @patch("app.controllers.auth_controller.get_collection")
    @patch("app.controllers.auth_controller.hash_password", return_value="hashed")
    async def test_register_success(self, _mock_hash, mock_get_coll):
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_collection.insert_one = AsyncMock()
        mock_get_coll.return_value = mock_collection

        payload = MagicMock(
            username="testuser", email="test@example.com", password="SecurePass1"
        )
        result = await self.controller.register(payload)

        assert result.username == "testuser"
        assert result.email == "test@example.com"
        mock_collection.insert_one.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.controllers.auth_controller.get_collection")
    async def test_register_duplicate_email(self, mock_get_coll):
        mock_collection = AsyncMock()
        # First find_one for email returns existing, second for username not called
        mock_collection.find_one = AsyncMock(return_value={"email": "test@example.com"})
        mock_get_coll.return_value = mock_collection

        payload = MagicMock(
            username="testuser", email="test@example.com", password="SecurePass1"
        )
        with pytest.raises(HTTPException) as exc_info:
            await self.controller.register(payload)
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    @patch("app.controllers.auth_controller.get_collection")
    async def test_register_duplicate_username(self, mock_get_coll):
        mock_collection = AsyncMock()
        # First find_one (email) returns None, second (username) returns existing
        mock_collection.find_one = AsyncMock(
            side_effect=[None, {"username": "testuser"}]
        )
        mock_get_coll.return_value = mock_collection

        payload = MagicMock(
            username="testuser", email="new@example.com", password="SecurePass1"
        )
        with pytest.raises(HTTPException) as exc_info:
            await self.controller.register(payload)
        assert exc_info.value.status_code == 409


class TestAuthControllerLogin:
    def setup_method(self):
        self.controller = AuthController()

    @pytest.mark.asyncio
    @patch("app.controllers.auth_controller.get_settings", return_value=MOCK_SETTINGS)
    @patch("app.controllers.auth_controller.create_access_token", return_value="jwt-token")
    @patch("app.controllers.auth_controller.verify_password", return_value=True)
    @patch("app.controllers.auth_controller.get_collection")
    async def test_login_success(self, mock_get_coll, _mock_verify, _mock_token, _mock_settings):
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(
            return_value={
                "_id": "user-123",
                "email": "test@example.com",
                "password_hash": "hashed",
                "is_active": True,
            }
        )
        mock_get_coll.return_value = mock_collection

        payload = MagicMock(email="test@example.com", password="SecurePass1")
        result = await self.controller.login(payload)

        assert result.access_token == "jwt-token"
        assert result.token_type == "bearer"

    @pytest.mark.asyncio
    @patch("app.controllers.auth_controller.verify_password", return_value=False)
    @patch("app.controllers.auth_controller.get_collection")
    async def test_login_wrong_password(self, mock_get_coll, _mock_verify):
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(
            return_value={
                "_id": "user-123",
                "email": "test@example.com",
                "password_hash": "hashed",
                "is_active": True,
            }
        )
        mock_get_coll.return_value = mock_collection

        payload = MagicMock(email="test@example.com", password="WrongPass1")
        with pytest.raises(HTTPException) as exc_info:
            await self.controller.login(payload)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch("app.controllers.auth_controller.get_collection")
    async def test_login_user_not_found(self, mock_get_coll):
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_get_coll.return_value = mock_collection

        payload = MagicMock(email="nobody@example.com", password="Pass1")
        with pytest.raises(HTTPException) as exc_info:
            await self.controller.login(payload)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch("app.controllers.auth_controller.verify_password", return_value=True)
    @patch("app.controllers.auth_controller.get_collection")
    async def test_login_deactivated_user(self, mock_get_coll, _mock_verify):
        mock_collection = AsyncMock()
        mock_collection.find_one = AsyncMock(
            return_value={
                "_id": "user-123",
                "email": "test@example.com",
                "password_hash": "hashed",
                "is_active": False,
            }
        )
        mock_get_coll.return_value = mock_collection

        payload = MagicMock(email="test@example.com", password="SecurePass1")
        with pytest.raises(HTTPException) as exc_info:
            await self.controller.login(payload)
        assert exc_info.value.status_code == 403
