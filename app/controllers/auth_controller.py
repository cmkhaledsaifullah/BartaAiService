import uuid
import logging

from fastapi import APIRouter, HTTPException, status

from app.auth.token import create_access_token, hash_password, verify_password
from app.config import get_settings
from app.constants import (
    COLLECTION_USERS,
    ERROR_EMAIL_ALREADY_EXISTS,
    ERROR_INVALID_CREDENTIALS,
    ERROR_USER_DEACTIVATED,
    ERROR_USERNAME_ALREADY_TAKEN,
)
from app.database.mongodb import get_collection
from app.models.user import (
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)

logger = logging.getLogger(__name__)


class AuthController:
    """Controller for user authentication and registration."""

    def __init__(self):
        self.router = APIRouter(prefix="/auth", tags=["authentication"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "/register",
            self.register,
            methods=["POST"],
            response_model=UserResponse,
            status_code=status.HTTP_201_CREATED,
        )
        self.router.add_api_route(
            "/login",
            self.login,
            methods=["POST"],
            response_model=TokenResponse,
        )

    async def register(self, payload: UserRegister):
        """Register a new user account."""
        collection = get_collection(COLLECTION_USERS)

        # Check if email already exists
        existing = await collection.find_one({"email": payload.email})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=ERROR_EMAIL_ALREADY_EXISTS,
            )

        # Check if username already exists
        existing_name = await collection.find_one({"username": payload.username})
        if existing_name:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=ERROR_USERNAME_ALREADY_TAKEN,
            )

        user_id = str(uuid.uuid4())
        user_doc = {
            "_id": user_id,
            "username": payload.username,
            "email": payload.email,
            "password_hash": hash_password(payload.password),
            "is_active": True,
        }
        await collection.insert_one(user_doc)
        logger.info("User registered: %s", payload.email)

        return UserResponse(
            id=user_id,
            username=payload.username,
            email=payload.email,
        )

    async def login(self, payload: UserLogin):
        """Authenticate a user and return a JWT access token."""
        collection = get_collection(COLLECTION_USERS)
        user = await collection.find_one({"email": payload.email.lower()})

        if not user or not verify_password(payload.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_INVALID_CREDENTIALS,
            )

        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=ERROR_USER_DEACTIVATED,
            )

        settings = get_settings()
        token = create_access_token(data={"sub": user["_id"], "email": user["email"]})

        return TokenResponse(
            access_token=token,
            expires_in=settings.jwt_access_token_expire_minutes * 60,
        )
