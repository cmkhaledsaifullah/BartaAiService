from fastapi import APIRouter

from app.constants import HEALTH_STATUS_HEALTHY, SERVICE_ID


class HealthController:
    """Controller for health check and service status endpoints."""

    def __init__(self):
        self.router = APIRouter(tags=["health"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/health", self.health_check, methods=["GET"])

    async def health_check(self):
        """Health check endpoint for load balancers and monitoring."""
        return {"status": HEALTH_STATUS_HEALTHY, "service": SERVICE_ID}
