from fastapi.testclient import TestClient
from igris.web.server import create_app


def test_api_status():
    client = TestClient(create_app())
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "provider" in data
    assert "model" in data