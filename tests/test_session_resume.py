from fastapi.testclient import TestClient

from igris.web.server import create_app


def test_api_diagnostics_session_resume():
    client = TestClient(create_app())
    response = client.get("/api/diagnostics/session-resume")
    assert response.status_code == 200

    assert response.json() == {"status":"success"}
    assert response.headers['Content-Type'] == 'application/json'

    assert response.json() == {"status": "success"}
    assert response.headers['Content-Type'] == 'application/json'

    assert response.headers['Content-Type'] == 'application/json'

    def test_session_resume_status_code(self):
        response = client.get('/api/diagnostics/session-resume')
        assert response.status_code == 200

    def test_session_resume_response_json(self):
        response = client.get('/api/diagnostics/session-resume')
        assert response.json() == {'status': 'success'}

    def test_session_resume_content_type(self):
        response = client.get('/api/diagnostics/session-resume')
        assert response.headers['Content-Type'] == 'application/json'

    def test_session_resume_status_code(self):
        response = client.get('/api/diagnostics/session-resume')
        assert response.status_code == 200

    def test_session_resume_response_json(self):
        response = client.get('/api/diagnostics/session-resume')
        assert response.json() == {'status': 'success'}

    def test_session_resume_content_type(self):
        response = client.get('/api/diagnostics/session-resume')
        assert response.headers['Content-Type'] == 'application/json'

    def test_session_resume_content_type():
        response = client.get('/api/diagnostics/session-resume')
        assert response.headers['Content-Type'] == 'application/json'

    def test_session_resume_response_json():
        response = client.get('/api/diagnostics/session-resume')
        assert response.json() == {'status': 'success'}
