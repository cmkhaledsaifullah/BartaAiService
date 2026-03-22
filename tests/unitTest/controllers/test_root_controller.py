import pytest
from unittest.mock import patch, MagicMock

from app.controllers.root_controller import RootController
from app.constants import SERVICE_NAME, SERVICE_VERSION, DOCS_DISABLED_MESSAGE


class TestRootController:
    def setup_method(self):
        self.controller = RootController()

    @pytest.mark.asyncio
    @patch("app.controllers.root_controller.get_settings")
    async def test_root_debug_mode(self, mock_settings):
        mock_settings.return_value = MagicMock(app_debug=True)
        result = await self.controller.root()
        assert result["service"] == SERVICE_NAME
        assert result["version"] == SERVICE_VERSION
        assert result["docs"] == "/docs"

    @pytest.mark.asyncio
    @patch("app.controllers.root_controller.get_settings")
    async def test_root_production_mode(self, mock_settings):
        mock_settings.return_value = MagicMock(app_debug=False)
        result = await self.controller.root()
        assert result["docs"] == DOCS_DISABLED_MESSAGE

    def test_router_has_root_route(self):
        routes = [r.path for r in self.controller.router.routes]
        assert "/" in routes
