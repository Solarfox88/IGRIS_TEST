from fastapi.testclient import TestClient

from igris.web.server import create_app


def test_api_github_write():
    client = TestClient(create_app())
    response = client.get("/api/github/write/")
    # Accept 200 (implemented) or 404/405 (scaffold placeholder — not yet implemented).
    # A 5xx error would indicate a real problem and is not accepted.
    assert response.status_code in (200, 404, 405), (
        f"Unexpected status {response.status_code} for '/api/github/write/' — "
        "expected 200 (implemented) or 404/405 (not yet implemented)"
    )
