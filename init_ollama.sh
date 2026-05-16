#!/usr/bin/env bash
# init_ollama.sh — 在 GPU 機器上首次部署時執行，將微調後的 GGUF 模型載入 Ollama
#
# 執行順序（在 GPU 機器上）：
#   1. 安裝 NVIDIA Container Toolkit（若尚未安裝）
#   2. docker compose up -d ollama        # 先啟動 Ollama
#   3. bash init_ollama.sh                # 載入模型
#   4. docker compose up -d api           # 再啟動 API server
#
# 或全部一起：
#   docker compose up -d
#   bash init_ollama.sh

set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
MODEL_TAG="${OLLAMA_MODEL:-trafficvision-gemma4}"
GGUF_DIR="$(cd "$(dirname "$0")" && pwd)/models/gguf"
GGUF_FILE="$GGUF_DIR/${MODEL_TAG}-q4_k_m.gguf"
MODELFILE="$GGUF_DIR/Modelfile"

echo "======================================================"
echo " TrafficVision — 初始化 Ollama 模型"
echo " Ollama URL: $OLLAMA_URL"
echo " 模型名稱:   $MODEL_TAG"
echo "======================================================"

# ── 等待 Ollama 就緒 ──────────────────────────────────────────────────────────
echo "等待 Ollama 服務就緒..."
for i in $(seq 1 30); do
    if curl -sf "$OLLAMA_URL/" > /dev/null 2>&1; then
        echo "  Ollama 已就緒。"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Ollama 未在 30 秒內啟動，請確認容器正在執行。"
        exit 1
    fi
    sleep 1
done

# ── 檢查模型是否已存在 ────────────────────────────────────────────────────────
if curl -sf "$OLLAMA_URL/api/tags" | grep -q "\"$MODEL_TAG\""; then
    echo "模型 '$MODEL_TAG' 已存在，跳過匯入。"
    echo "  若要重新匯入，請先執行：curl -X DELETE $OLLAMA_URL/api/delete -d '{\"name\":\"$MODEL_TAG\"}'"
    exit 0
fi

# ── 確認 GGUF 檔案存在 ────────────────────────────────────────────────────────
if [ ! -f "$GGUF_FILE" ]; then
    echo "ERROR: 找不到 GGUF 檔案：$GGUF_FILE"
    echo ""
    echo "請先完成微調流程："
    echo "  1. python tools/generate_finetune_dataset.py"
    echo "  2. python tools/finetune_gemma.py"
    echo "  3. bash tools/export_to_gguf.sh"
    echo ""
    echo "或使用通用 Gemma 3 測試："
    echo "  docker exec trafficvision-ollama ollama pull gemma3:12b"
    echo "  OLLAMA_MODEL=gemma3:12b docker compose up -d api"
    exit 1
fi

# ── 在容器內建立 Modelfile 並匯入 ─────────────────────────────────────────────
echo "正在將模型匯入 Ollama 容器..."

# 複製 GGUF 和 Modelfile 進容器（透過已掛載的 volume 直接存取）
if [ -f "$MODELFILE" ]; then
    cd "$GGUF_DIR"
    # 容器的 /models/gguf volume 對應 ./models/gguf
    docker exec trafficvision-ollama ollama create "$MODEL_TAG" \
        -f "/models/gguf/Modelfile"
else
    # Fallback: 直接用 ollama pull 替代方案
    echo "WARNING: 找不到 Modelfile，嘗試直接匯入 GGUF..."
    docker exec trafficvision-ollama ollama create "$MODEL_TAG" \
        -f - <<MFEOF
FROM /models/gguf/${MODEL_TAG}-q4_k_m.gguf
SYSTEM "你是 TrafficVision AI 助理，專門分析台北市北科大周邊路網的即時車流、GRU 預測與號誌優化結果。請使用繁體中文回答，數據需引用具體數值。"
PARAMETER temperature 0.3
PARAMETER num_ctx 2048
MFEOF
fi

echo ""
echo "======================================================"
echo " 完成！模型 '$MODEL_TAG' 已就緒。"
echo ""
echo " 測試："
echo "   curl http://localhost:8000/api/status"
echo "   curl -X POST http://localhost:8000/api/chat \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"message\": \"目前交通狀況如何？\"}'"
echo "======================================================"
