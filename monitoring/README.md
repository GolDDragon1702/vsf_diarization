# Giám sát (Observability)

API expose metrics theo chuẩn **Prometheus** tại `GET /metrics/prometheus`, kèm
JSON gọn tại `GET /metrics` (frontend dùng để vẽ gauge). Mọi request HTTP/WS được
**log có cấu trúc** (method, path, status, thời gian).

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

## Chạy Prometheus + Grafana

```bash
vsf-serve --host 0.0.0.0                              # API + metrics ở :8000
docker compose -f monitoring/docker-compose.yml up    # Prometheus :9090 · Grafana :3000
```

Grafana (admin/admin) → Add data source → Prometheus → `http://prometheus:9090`.
Ví dụ query: `rate(vsf_requests_total[1m])`, `histogram_quantile(0.95, vsf_rtf_bucket)`.

> `prometheus_client` là tuỳ chọn; thiếu nó `/metrics/prometheus` trả 501 nhưng API
> và dashboard JSON `/metrics` vẫn chạy bình thường.
