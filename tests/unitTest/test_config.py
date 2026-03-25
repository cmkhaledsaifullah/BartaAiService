import pytest
from unittest.mock import patch, MagicMock

from app.config import Settings, get_settings


class TestGetSettings:
    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://localhost:27017",
        "JWT_SECRET_KEY": "test-secret",
    })
    def test_get_settings_returns_settings_instance(self):
        from app.config import get_settings, Settings
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)
        get_settings.cache_clear()

class TestSettingsProperties:
    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://localhost:27017",
        "JWT_SECRET_KEY": "test-secret",
    })
    def test_cors_origin_list(self):
        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            jwt_secret_key="test-secret",
            cors_origins="http://localhost:3000, http://localhost:5173",
        )
        assert settings.cors_origin_list == ["http://localhost:3000", "http://localhost:5173"]

    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://localhost:27017",
        "JWT_SECRET_KEY": "test-secret",
    })
    def test_cors_origin_list_single(self):
        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            jwt_secret_key="test-secret",
            cors_origins="http://localhost:3000",
        )
        assert settings.cors_origin_list == ["http://localhost:3000"]

    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://localhost:27017",
        "JWT_SECRET_KEY": "test-secret",
    })
    def test_cors_allow_methods_list(self):
        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            jwt_secret_key="test-secret",
            cors_allow_methods="GET,POST,OPTIONS",
        )
        assert settings.cors_allow_methods_list == ["GET", "POST", "OPTIONS"]

    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://localhost:27017",
        "JWT_SECRET_KEY": "test-secret",
    })
    def test_cors_allow_headers_list(self):
        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            jwt_secret_key="test-secret",
            cors_allow_headers="Authorization, Content-Type",
        )
        assert settings.cors_allow_headers_list == ["Authorization", "Content-Type"]

    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://localhost:27017",
        "JWT_SECRET_KEY": "test-secret",
    })
    def test_default_values(self):
        settings = Settings(
            mongodb_uri="mongodb://localhost:27017",
            jwt_secret_key="test-secret",
        )
        assert settings.mongodb_db_name == "bartaAi"
        assert settings.llm_provider == "openai"
        assert settings.llm_model == "gpt-4o"
        assert settings.embedding_provider == "openai"
        assert settings.jwt_algorithm == "HS256"
        assert settings.jwt_access_token_expire_minutes == 60
        assert settings.rate_limit_enabled is True
        assert settings.app_env == "development"
        assert settings.app_debug is False
