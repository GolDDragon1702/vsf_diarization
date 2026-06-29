"""Test API endpoint không cần model: /health, /metrics."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from vsf_diarization.serve.api import create_app


def test_liveness_always_ok():
    client = TestClient(create_app(mount_demo=False))
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"             # liveness không phụ thuộc model


def test_readiness_503_before_model_load():
    client = TestClient(create_app(mount_demo=False))
    r = client.get("/ready")
    body = r.json()
    assert r.status_code == 503                       # chưa load model → chưa ready
    assert body["ready"] is False
    assert body["models_loaded"] is False
    assert {"gpu_ok", "device"} <= set(body)


def test_metrics_no_model_load():
    client = TestClient(create_app(mount_demo=False))
    m = client.get("/metrics")
    assert m.status_code == 200
    assert {"uptime_s", "device", "avg_rtf", "rest_requests"} <= set(m.json())


def test_frontend_served():
    client = TestClient(create_app(mount_demo=False))
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/app/"
    page = client.get("/app/")
    assert page.status_code == 200
    assert "VSF Diarization" in page.text


def test_transcribe_input_validation():
    """Validate xảy ra TRƯỚC khi load model (không cần GPU/model)."""
    client = TestClient(create_app(mount_demo=False))

    # file rỗng → 400
    r = client.post("/transcribe", files={"file": ("a.wav", b"", "audio/wav")})
    assert r.status_code == 400

    # bytes rác (không phải audio) → 400, không crash, không load model
    r = client.post("/transcribe", files={"file": ("a.wav", b"not audio data", "audio/wav")})
    assert r.status_code == 400
    assert "đọc được audio" in r.json()["detail"]

    # num_speakers ngoài [1,10] → 422 (Query constraint)
    r = client.post("/transcribe?num_speakers=99",
                    files={"file": ("a.wav", b"x", "audio/wav")})
    assert r.status_code == 422

    assert client.get("/ready").json()["models_loaded"] is False   # vẫn chưa load


def test_prometheus_endpoint():
    client = TestClient(create_app(mount_demo=False))
    r = client.get("/metrics/prometheus")
    # 200 nếu prometheus_client có; 501 nếu chưa cài (vẫn không lỗi server)
    assert r.status_code in (200, 501)
    if r.status_code == 200:
        assert "vsf_requests_total" in r.text
