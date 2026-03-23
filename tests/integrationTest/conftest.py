import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch, AsyncMock

from app.main import app


@pytest_asyncio.fixture
async def client():
    """Async test client that skips real MongoDB connections."""
    with patch("app.main.connect_to_mongodb", new_callable=AsyncMock):
        with patch("app.main.ensure_indexes", new_callable=AsyncMock):
            with patch("app.main.ensure_click_log_indexes", new_callable=AsyncMock):
                with patch("app.main.close_mongodb_connection", new_callable=AsyncMock):
                    transport = ASGITransport(app=app)
                    async with AsyncClient(
                        transport=transport, base_url="http://test"
                    ) as ac:
                        yield ac
