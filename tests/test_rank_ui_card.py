from pathlib import Path

from fastapi.testclient import TestClient

from igris.web.server import create_app
from igris.web import server as server_module


def test_rank_ui_card_endpoint_available():
    client = TestClient(create_app())
    response = client.get("/api/rank/ui-card")

    assert response.status_code == 200


def test_rank_ui_card_route_is_defined_once_in_create_app():
    # After #725 refactor, routes live in igris/web/routers/routes_*.py
    # using @router.get instead of @app.get in server.py.
    routers_dir = Path(server_module.__file__).parent / "routers"
    route = "@router.get('/api/rank/ui-card')"
    total_count = sum(
        f.read_text().count(route)
        for f in routers_dir.glob("routes_*.py")
    )
    assert total_count == 1, (
        f"Expected exactly 1 definition of {route!r} across router modules, "
        f"found {total_count}"
    )

def test_rank_ui_card_response():
    client = TestClient(create_app())
    response = client.get('/api/rank/ui-card')
    assert response.status_code == 200
    data = response.json()
    assert 'app' in data
    assert 'rank' in data
    assert 'status' in data
    assert 'capability' in data
