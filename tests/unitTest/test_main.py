import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestAppCreation:
    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://localhost:27017",
        "JWT_SECRET_KEY": "test-secret",
    })
    def test_app_is_created(self):
        from importlib import reload
        import app.config
        app.config.get_settings.cache_clear()
        import app.main
        reload(app.main)
        assert app.main.app is not None
        assert app.main.app.title == "Barta AI Service"
        app.config.get_settings.cache_clear()

    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://localhost:27017",
        "JWT_SECRET_KEY": "test-secret",
    })
    def test_routes_are_registered(self):
        from importlib import reload
        import app.config
        app.config.get_settings.cache_clear()
        import app.main
        reload(app.main)
        paths = [route.path for route in app.main.app.routes]
        assert "/" in paths
        assert "/api/v1/health" in paths
        app.config.get_settings.cache_clear()


class TestLifecycleEvents:
    @pytest.mark.asyncio
    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://localhost:27017",
        "JWT_SECRET_KEY": "test-secret",
    })
    async def test_startup(self):
        import app.config
        app.config.get_settings.cache_clear()

        with patch("app.database.mongodb.get_settings") as mock_db_settings, \
             patch("app.database.mongodb.AsyncIOMotorClient") as mock_client_cls, \
             patch("app.database.vector_store.get_collection") as mock_coll, \
             patch("app.services.news_service.get_collection") as mock_news_coll:
            mock_db_settings.return_value = MagicMock(
                mongodb_uri="mongodb://localhost:27017",
                mongodb_db_name="test_db",
                mongodb_tls_cert_key_file="",
            )
            mock_client = MagicMock()
            mock_client.admin.command = AsyncMock(return_value={"ok": 1})
            mock_client.__getitem__ = MagicMock(return_value=MagicMock())
            mock_client_cls.return_value = mock_client

            mock_collection = MagicMock()
            mock_collection.create_index = AsyncMock()
            mock_coll.return_value = mock_collection
            mock_news_coll.return_value = mock_collection

            from app.main import startup
            await startup()
            mock_client.admin.command.assert_called_once_with("ping")

        app.config.get_settings.cache_clear()

    @pytest.mark.asyncio
    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://localhost:27017",
        "JWT_SECRET_KEY": "test-secret",
    })
    async def test_shutdown(self):
        import app.config
        app.config.get_settings.cache_clear()
        import app.database.mongodb as mongodb_module

        mock_client = MagicMock()
        mongodb_module._client = mock_client
        mongodb_module._database = MagicMock()

        from app.main import shutdown
        await shutdown()

        mock_client.close.assert_called_once()
        assert mongodb_module._client is None
        app.config.get_settings.cache_clear()


class TestExceptionHandler:
    @pytest.mark.asyncio
    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://localhost:27017",
        "JWT_SECRET_KEY": "test-secret",
    })
    async def test_global_exception_handler(self):
        from importlib import reload
        import app.config
        app.config.get_settings.cache_clear()
        import app.main
        reload(app.main)

        from fastapi import Request
        mock_request = MagicMock(spec=Request)
        response = await app.main.global_exception_handler(
            mock_request, Exception("test error")
        )
        assert response.status_code == 500
        app.config.get_settings.cache_clear()
