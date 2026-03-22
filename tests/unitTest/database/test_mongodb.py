import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.database.mongodb import (
    connect_to_mongodb,
    close_mongodb_connection,
    get_database,
    get_collection,
)
import app.database.mongodb as mongodb_module


class TestConnectToMongodb:
    def teardown_method(self):
        mongodb_module._client = None
        mongodb_module._database = None

    @pytest.mark.asyncio
    @patch("app.database.mongodb.get_settings")
    @patch("app.database.mongodb.AsyncIOMotorClient")
    async def test_connect_success(self, mock_client_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            mongodb_uri="mongodb://localhost:27017",
            mongodb_db_name="test_db",
        )
        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        mock_client.__getitem__ = MagicMock(return_value=MagicMock())
        mock_client_cls.return_value = mock_client

        await connect_to_mongodb()

        mock_client_cls.assert_called_once_with("mongodb://localhost:27017")
        mock_client.admin.command.assert_called_once_with("ping")
        assert mongodb_module._client is not None
        assert mongodb_module._database is not None


class TestCloseMongodb:
    def teardown_method(self):
        mongodb_module._client = None
        mongodb_module._database = None

    @pytest.mark.asyncio
    async def test_close_when_connected(self):
        mock_client = MagicMock()
        mongodb_module._client = mock_client
        mongodb_module._database = MagicMock()

        await close_mongodb_connection()

        mock_client.close.assert_called_once()
        assert mongodb_module._client is None
        assert mongodb_module._database is None

    @pytest.mark.asyncio
    async def test_close_when_not_connected(self):
        mongodb_module._client = None
        mongodb_module._database = None

        await close_mongodb_connection()  # should not raise


class TestGetDatabase:
    def teardown_method(self):
        mongodb_module._client = None
        mongodb_module._database = None

    def test_returns_database_when_connected(self):
        mock_db = MagicMock()
        mongodb_module._database = mock_db
        assert get_database() is mock_db

    def test_raises_when_not_connected(self):
        mongodb_module._database = None
        with pytest.raises(RuntimeError, match="MongoDB is not connected"):
            get_database()


class TestGetCollection:
    def teardown_method(self):
        mongodb_module._client = None
        mongodb_module._database = None

    def test_returns_collection(self):
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        mongodb_module._database = mock_db

        result = get_collection("users")
        mock_db.__getitem__.assert_called_once_with("users")
        assert result is mock_collection

    def test_raises_when_not_connected(self):
        mongodb_module._database = None
        with pytest.raises(RuntimeError):
            get_collection("users")
