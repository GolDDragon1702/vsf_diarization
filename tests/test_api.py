"""Test API endpoint không cần model: /health, /metrics."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from vsf_diarization.serve.api import create_app


def test_health_and_metrics_no_model_load():
    client = TestClient(create_app(mount_demo=False))

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["models_loaded"] is False        # chưa gọi /transcribe → chưa load

    m = client.get("/metrics")
    assert m.status_code == 200
    body = m.json()
    assert {"uptime_s", "device", "avg_rtf", "rest_requests"} <= set(body)


def test_frontend_served():
    client = TestClient(create_app(mount_demo=False))
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/app/"
    page = client.get("/app/")
    assert page.status_code == 200
    assert "VSF Diarization" in page.text


def test_prometheus_endpoint():
    client = TestClient(create_app(mount_demo=False))
    r = client.get("/metrics/prometheus")
    # 200 nếu prometheus_client có; 501 nếu chưa cài (vẫn không lỗi server)
    assert r.status_code in (200, 501)
    if r.status_code == 200:
        assert "vsf_requests_total" in r.text
