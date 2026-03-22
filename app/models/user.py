from pydantic import BaseModel, Field, field_validator
import re

from app.constants import (
    ERROR_INVALID_EMAIL_FORMAT,
    ERROR_PASSWORD_NO_DIGIT,
    ERROR_PASSWORD_NO_LOWERCASE,
    ERROR_PASSWORD_NO_UPPERCASE,
)


class UserRegister(BaseModel):
    """Request body for user registration."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError(ERROR_INVALID_EMAIL_FORMAT)
        return v.lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError(ERROR_PASSWORD_NO_UPPERCASE)
        if not re.search(r"[a-z]", v):
            raise ValueError(ERROR_PASSWORD_NO_LOWERCASE)
        if not re.search(r"\d", v):
            raise ValueError(ERROR_PASSWORD_NO_DIGIT)
        return v


class UserLogin(BaseModel):
    """Request body for user login."""
    email: str
    password: str


class TokenResponse(BaseModel):
    """Response body containing the access token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    """Public user information in responses."""
    id: str
    username: str
    email: str
    is_active: bool = True
