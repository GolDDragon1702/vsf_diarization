# Giám sát (Observability)

API expose metrics theo chuẩn **Prometheus** tại `GET /metrics/prometheus`, kèm
JSON gọn tại `GET /metrics` (frontend dùng để vẽ gauge). Mọi request HTTP/WS được
**log có cấu trúc** (method, path, status, thời gian).

Probe sức khỏe: `GET /health` = **liveness** (luôn 200), `GET /ready` = **readiness**
(model đã nạp + GPU sống → 200, ngược lại 503 — hợp với liveness/readinessProbe k8s).

## Metrics

| Metric | Loại | Ý nghĩa |
|--------|------|---------|
| `vsf_requests_total{endpoint,status}` | Counter | Số request theo endpoint + HTTP status |
| `vsf_request_seconds{endpoint}` | Histogram | Độ trễ xử lý |
| `vsf_rtf{endpoint}` | Histogram | Real-time factor (proc/audio) |
| `vsf_audio_seconds_total` | Counter | Tổng audio đã xử lý |
| `vsf_turns_total` | Counter | Tổng speaker-turn phát ra |
| `vsf_active_ws_sessions` | Gauge | Phiên WebSocket đang mở |
| `vsf_model_load_seconds` | Gauge | Thời gian load model |
| `vsf_gpu_memory_bytes{type}` | Gauge | VRAM allocated/reserved |

## Full-stack 1 lệnh (App + Prometheus + Grafana)

```bash
export HF_TOKEN=hf_xxx                                # cần token pyannote
docker compose -f monitoring/docker-compose.yml up --build
```

| Service | URL |
|---------|-----|
| App (web UI + API) | http://localhost:8000/ |
| Prometheus | http://localhost:9090 |
| Grafana (admin/admin) | http://localhost:3000 |

Grafana **tự nạp** datasource Prometheus + dashboard **"VSF Diarization"**
([grafana/dashboards/vsf-diarization.json](grafana/dashboards/vsf-diarization.json)) — không cần cấu hình tay.
Panel: Avg RTF, p95 latency/endpoint, request rate, active WS, GPU memory, turns/s.

> Nếu chỉ chạy API trên host (không qua compose): `vsf-serve --host 0.0.0.0` rồi sửa
> target trong [prometheus.yml](prometheus.yml) thành `host.docker.internal:8000`.

> `prometheus_client` là tuỳ chọn; thiếu nó `/metrics/prometheus` trả 501 nhưng API
> và dashboard JSON `/metrics` vẫn chạy bình thường.
