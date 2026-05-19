from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app
from tests.helpers import assert_response_contract


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.integration
def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_analyze_invalid_file_returns_400(client: TestClient) -> None:
    response = client.post(
        "/analyze",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
        data={"profile_id": "pytest"},
    )

    assert response.status_code == 400


@pytest.mark.integration
@pytest.mark.slow
def test_analyze_endpoint_returns_stable_response_contract(
    client: TestClient,
    pass_image: Path,
    response_contract_keys: set[str],
) -> None:
    response = client.post(
        "/analyze",
        files={
            "file": (
                pass_image.name,
                pass_image.read_bytes(),
                "image/jpeg",
            )
        },
        data={"profile_id": "pytest-api"},
    )

    assert response.status_code == 200

    body = response.json()
    assert_response_contract(body, response_contract_keys)
    assert body["profileId"] == "pytest-api"
    assert body["verdict"] == "PASS"
    assert body["summary"]["decision"] == "PASS"
    assert body["errors"] == []
