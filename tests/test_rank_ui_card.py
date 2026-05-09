from pathlib import Path

from fastapi.testclient import TestClient

from igris.web.server import create_app
from igris.web import server as server_module


def test_rank_ui_card_endpoint_available():
    client = TestClient(create_app())
    response = client.get("/api/rank/ui-card")

    assert response.status_code == 200


def test_rank_ui_card_route_is_defined_once_in_create_app():
    source = Path(server_module.__file__).read_text()
    route = "@app.get('/api/rank/ui-card')"

    assert source.count(route) == 1
    assert source.index(route) < source.index("def run_app")
