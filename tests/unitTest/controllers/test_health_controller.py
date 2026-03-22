import pytest

from app.controllers.health_controller import HealthController
from app.constants import HEALTH_STATUS_HEALTHY, SERVICE_ID


class TestHealthController:
    def setup_method(self):
        self.controller = HealthController()

    @pytest.mark.asyncio
    async def test_health_check(self):
        result = await self.controller.health_check()
        assert result["status"] == HEALTH_STATUS_HEALTHY
        assert result["service"] == SERVICE_ID

    def test_router_has_health_route(self):
        routes = [r.path for r in self.controller.router.routes]
        assert "/health" in routes
