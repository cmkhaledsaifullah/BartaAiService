from fastapi import APIRouter

from app.config import get_settings
from app.constants import DOCS_DISABLED_MESSAGE, SERVICE_NAME, SERVICE_VERSION


class RootController:
    """Controller for the root/info endpoint."""

    def __init__(self):
        self.router = APIRouter(tags=["root"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/", self.root, methods=["GET"])

    async def root(self):
        """Service information endpoint."""
        settings = get_settings()
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "docs": "/docs" if settings.app_debug else DOCS_DISABLED_MESSAGE,
        }
