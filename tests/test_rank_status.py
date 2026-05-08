import pytest
from fastapi.testclient import TestClient
from igris.web.server import create_app

app = create_app()


def test_get_rank_status():
    client = TestClient(app)
    response = client.get('/api/rank/status')
    assert response.status_code == 200
    assert response.json() == {'rank': 'A', 'status': 'ok', 'agent': 'IGRIS_GPT'}