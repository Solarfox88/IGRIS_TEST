import pytest
from fastapi.testclient import TestClient
from igris.web.server import create_app

app = create_app()

# Test cases for Rank S dashboard

def test_rank_s_dashboard():
    response = TestClient(app).get('/api/rank/s-dashboard')
    assert response.status_code == 200
    assert response.json() == {'app':'IGRIS_GPT','rank':'S','status':'ok','capability':'end-to-end-supervised','checks':{'backend':True,'ui':True,'tests':True,'workflow':True}}

# Operational note: Added visibility for Rank S dashboard endpoint to ensure UI/dashboard reflects the new API.
