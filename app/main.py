import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.constants import (
    API_PREFIX,
    ERROR_INTERNAL_SERVER,
    SERVICE_DESCRIPTION,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from app.database.mongodb import connect_to_mongodb, close_mongodb_connection
from app.database.vector_store import ensure_indexes
from app.controllers.root_controller import RootController
from app.controllers.health_controller import HealthController
from app.controllers.auth_controller import AuthController
from app.controllers.chat_controller import ChatController
from app.services.news_service import ensure_click_log_indexes

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=SERVICE_NAME,
    description=SERVICE_DESCRIPTION,
    version=SERVICE_VERSION,
    docs_url="/docs" if settings.app_debug else None,
    redoc_url="/redoc" if settings.app_debug else None,
)

# Rate limiting (settings-driven)
if settings.rate_limit_enabled:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS (settings-driven)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods_list,
    allow_headers=settings.cors_allow_headers_list,
)


# --- Lifecycle Events ---

@app.on_event("startup")
async def startup():
    logger.info("Starting Barta AI Service...")
    await connect_to_mongodb()
    await ensure_indexes()
    await ensure_click_log_indexes()
    logger.info("Barta AI Service started successfully")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down Barta AI Service...")
    await close_mongodb_connection()


# --- Global exception handler ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": ERROR_INTERNAL_SERVER},
    )


# --- Register Controllers ---

root_controller = RootController()
health_controller = HealthController()
auth_controller = AuthController()
chat_controller = ChatController()

app.include_router(root_controller.router)
app.include_router(health_controller.router, prefix=API_PREFIX)
app.include_router(auth_controller.router, prefix=API_PREFIX)
app.include_router(chat_controller.router, prefix=API_PREFIX)
