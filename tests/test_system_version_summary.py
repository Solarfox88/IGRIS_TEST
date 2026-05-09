import pytest
from fastapi.testclient import TestClient
from igris.web.server import create_app

app = create_app()


def test_version_summary():
    client = TestClient(app)
    response = client.get('/api/system/version-summary')
    assert response.status_code == 200
    assert response.json() == {'app': 'IGRIS_GPT', 'rank': 'A-generalization', 'status': 'ok'}