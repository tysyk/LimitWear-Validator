from __future__ import annotations

import unittest

try:
    from fastapi.testclient import TestClient
    from main import app
except Exception:  # pragma: no cover - depends on local runtime
    TestClient = None
    app = None


@unittest.skipUnless(TestClient is not None and app is not None, "FastAPI runtime is unavailable")
class HealthApiTests(unittest.TestCase):
    def test_health_endpoint_returns_ok(self):
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
