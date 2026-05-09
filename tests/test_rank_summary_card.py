from fastapi.testclient import TestClient
from igris.web.server import create_app

def test_rank_summary_card():
    client = TestClient(create_app())
    response = client.get('/api/rank/summary-card')
    assert response.status_code == 200
    assert response.json() == {"app":"IGRIS_GPT","rank":"A+","status":"ok","capability":"multi-file-supervised"}
