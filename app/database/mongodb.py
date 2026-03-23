import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import get_settings
from app.constants import ERROR_MONGODB_NOT_CONNECTED

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None


async def connect_to_mongodb() -> None:
    """Initialize MongoDB connection on application startup."""
    global _client, _database
    settings = get_settings()
    client_kwargs: dict = {}
    if settings.mongodb_tls_cert_key_file:
        client_kwargs["tls"] = True
        client_kwargs["tlsCertificateKeyFile"] = settings.mongodb_tls_cert_key_file
    _client = AsyncIOMotorClient(settings.mongodb_uri, **client_kwargs)
    _database = _client[settings.mongodb_db_name]
    # Verify connectivity
    await _client.admin.command("ping")
    logger.info("Connected to MongoDB: %s", settings.mongodb_db_name)


async def close_mongodb_connection() -> None:
    """Close MongoDB connection on application shutdown."""
    global _client, _database
    if _client:
        _client.close()
        _client = None
        _database = None
        logger.info("Closed MongoDB connection")


def get_database() -> AsyncIOMotorDatabase:
    """Get the database instance. Must be called after connect_to_mongodb()."""
    if _database is None:
        raise RuntimeError(ERROR_MONGODB_NOT_CONNECTED)
    return _database


def get_collection(name: str):
    """Get a MongoDB collection by name."""
    return get_database()[name]
