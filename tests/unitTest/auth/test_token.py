import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

from app.auth.token import (
    verify_password,
    hash_password,
    create_access_token,
    decode_access_token,
)


MOCK_SETTINGS = MagicMock(
    jwt_secret_key="test-secret-key-for-unit-tests",
    jwt_algorithm="HS256",
    jwt_access_token_expire_minutes=60,
)


class TestHashPassword:
    def test_returns_hashed_string(self):
        hashed = hash_password("SecurePass1")
        assert hashed != "SecurePass1"
        assert isinstance(hashed, str)

    def test_different_passwords_produce_different_hashes(self):
        h1 = hash_password("PasswordA1")
        h2 = hash_password("PasswordB2")
        assert h1 != h2

    def test_same_password_produces_different_hashes_due_to_salt(self):
        h1 = hash_password("SecurePass1")
        h2 = hash_password("SecurePass1")
        assert h1 != h2


class TestVerifyPassword:
    def test_correct_password_returns_true(self):
        hashed = hash_password("SecurePass1")
        assert verify_password("SecurePass1", hashed) is True

    def test_wrong_password_returns_false(self):
        hashed = hash_password("SecurePass1")
        assert verify_password("WrongPass1", hashed) is False

    def test_empty_password_returns_false(self):
        hashed = hash_password("SecurePass1")
        assert verify_password("", hashed) is False


class TestCreateAccessToken:
    @patch("app.auth.token.get_settings", return_value=MOCK_SETTINGS)
    def test_returns_string_token(self, _mock):
        token = create_access_token(data={"sub": "user-123"})
        assert isinstance(token, str)
        assert len(token) > 0

    @patch("app.auth.token.get_settings", return_value=MOCK_SETTINGS)
    def test_token_contains_payload_data(self, _mock):
        token = create_access_token(data={"sub": "user-123", "email": "a@b.com"})
        decoded = decode_access_token(token)
        assert decoded["sub"] == "user-123"
        assert decoded["email"] == "a@b.com"

    @patch("app.auth.token.get_settings", return_value=MOCK_SETTINGS)
    def test_token_contains_exp_and_iat(self, _mock):
        token = create_access_token(data={"sub": "user-123"})
        decoded = decode_access_token(token)
        assert "exp" in decoded
        assert "iat" in decoded

    @patch("app.auth.token.get_settings", return_value=MOCK_SETTINGS)
    def test_custom_expiry(self, _mock):
        delta = timedelta(minutes=5)
        token = create_access_token(data={"sub": "user-123"}, expires_delta=delta)
        decoded = decode_access_token(token)
        iat = decoded["iat"]
        exp = decoded["exp"]
        assert (exp - iat) == 300  # 5 minutes in seconds


class TestDecodeAccessToken:
    @patch("app.auth.token.get_settings", return_value=MOCK_SETTINGS)
    def test_decode_valid_token(self, _mock):
        token = create_access_token(data={"sub": "user-123"})
        decoded = decode_access_token(token)
        assert decoded["sub"] == "user-123"

    @patch("app.auth.token.get_settings", return_value=MOCK_SETTINGS)
    def test_decode_invalid_token_raises(self, _mock):
        import jwt

        with pytest.raises(jwt.InvalidTokenError):
            decode_access_token("invalid.token.string")

    @patch("app.auth.token.get_settings", return_value=MOCK_SETTINGS)
    def test_decode_expired_token_raises(self, _mock):
        import jwt

        token = create_access_token(
            data={"sub": "user-123"},
            expires_delta=timedelta(seconds=-1),
        )
        with pytest.raises(jwt.ExpiredSignatureError):
            decode_access_token(token)
