# ── TrafficVision API Server ──────────────────────────────────────────────────
# Python 3.11 slim image; no SUMO required for the API-only container.
FROM python:3.11-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先複製依賴清單以利用 Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製整個專案
COPY . .

# 建立資料目錄（掛載 volume 後會覆蓋，但確保目錄存在）
RUN mkdir -p \
    "TrafficVision Design System/data/trafficData" \
    "TrafficVision Design System/data/runtime_data" \
    "TrafficVision Design System/data" \
    data

# P7: drop root inside the container — audit L1. With ./data bind-mounted,
# anything writing as root inside leaves root-owned files on the host that
# the host user can't clean up without sudo. UID 1000 matches the typical
# first non-root host user.
RUN useradd -m -u 1000 trafficvision && \
    chown -R trafficvision:trafficvision /app

USER trafficvision

EXPOSE 8000

# health check（依賴 /api/status 端點）
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

CMD ["python", "TrafficVision Design System/serve_api.py"]
