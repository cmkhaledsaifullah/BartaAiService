import pytest
from pydantic import ValidationError

from app.models.user import UserRegister, UserLogin, TokenResponse, UserResponse


class TestUserRegister:
    def test_valid_registration(self):
        user = UserRegister(
            username="testuser", email="test@example.com", password="SecurePass1"
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"

    def test_email_normalized_to_lowercase(self):
        user = UserRegister(
            username="testuser", email="Test@Example.COM", password="SecurePass1"
        )
        assert user.email == "test@example.com"

    def test_invalid_email_format(self):
        with pytest.raises(ValidationError, match="email"):
            UserRegister(username="testuser", email="not-an-email", password="SecurePass1")

    def test_password_no_uppercase(self):
        with pytest.raises(ValidationError, match="uppercase"):
            UserRegister(username="testuser", email="test@example.com", password="nouppercas1")

    def test_password_no_lowercase(self):
        with pytest.raises(ValidationError, match="lowercase"):
            UserRegister(username="testuser", email="test@example.com", password="NOLOWERCASE1")

    def test_password_no_digit(self):
        with pytest.raises(ValidationError, match="digit"):
            UserRegister(username="testuser", email="test@example.com", password="NoDigitHere")

    def test_password_too_short(self):
        with pytest.raises(ValidationError):
            UserRegister(username="testuser", email="test@example.com", password="Sh1")

    def test_username_too_short(self):
        with pytest.raises(ValidationError):
            UserRegister(username="ab", email="test@example.com", password="SecurePass1")

    def test_username_too_long(self):
        with pytest.raises(ValidationError):
            UserRegister(
                username="a" * 51, email="test@example.com", password="SecurePass1"
            )


class TestUserLogin:
    def test_valid_login(self):
        login = UserLogin(email="test@example.com", password="SecurePass1")
        assert login.email == "test@example.com"

    def test_missing_email(self):
        with pytest.raises(ValidationError):
            UserLogin(password="SecurePass1")

    def test_missing_password(self):
        with pytest.raises(ValidationError):
            UserLogin(email="test@example.com")


class TestTokenResponse:
    def test_default_token_type(self):
        token = TokenResponse(access_token="jwt-token", expires_in=3600)
        assert token.token_type == "bearer"
        assert token.access_token == "jwt-token"
        assert token.expires_in == 3600


class TestUserResponse:
    def test_defaults(self):
        user = UserResponse(id="u1", username="test", email="test@example.com")
        assert user.is_active is True

    def test_inactive_user(self):
        user = UserResponse(
            id="u1", username="test", email="test@example.com", is_active=False
        )
        assert user.is_active is False
