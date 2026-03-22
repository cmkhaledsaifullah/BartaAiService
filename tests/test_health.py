import pytest

from app.constants import HEALTH_STATUS_HEALTHY, SERVICE_ID, SERVICE_NAME


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == HEALTH_STATUS_HEALTHY
    assert data["service"] == SERVICE_ID


@pytest.mark.asyncio
async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == SERVICE_NAME
