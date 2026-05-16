#!/usr/bin/env bash
# export_to_gguf.sh — 合併 LoRA、轉 GGUF、匯入 Ollama
#
# 先決條件：
#   1. pip install unsloth
#   2. git clone https://github.com/ggerganov/llama.cpp.git $HOME/llama.cpp
#      pip install -r $HOME/llama.cpp/requirements.txt
#   3. ollama 已安裝並在背景執行（ollama serve）
#
# 環境變數（可覆蓋預設值）：
#   LLAMA_CPP_PATH=/path/to/llama.cpp   （預設：$HOME/llama.cpp）
#   OLLAMA_MODEL_NAME=trafficvision-gemma4

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADAPTER_DIR="$ROOT/models/trafficvision-gemma4"
MERGED_DIR="$ROOT/models/trafficvision-gemma4-merged"
GGUF_DIR="$ROOT/models/gguf"
LLAMA_CPP="${LLAMA_CPP_PATH:-$HOME/llama.cpp}"
MODEL_TAG="${OLLAMA_MODEL_NAME:-trafficvision-gemma4}"
GGUF_FILE="$GGUF_DIR/${MODEL_TAG}-q4_k_m.gguf"

echo "======================================================"
echo " TrafficVision Gemma 4 — GGUF Export & Ollama Import"
echo "======================================================"
echo "  Adapter dir : $ADAPTER_DIR"
echo "  llama.cpp   : $LLAMA_CPP"
echo "  Ollama tag  : $MODEL_TAG"
echo ""

# 確認 adapter 目錄存在
if [ ! -d "$ADAPTER_DIR" ]; then
    echo "ERROR: 找不到 $ADAPTER_DIR"
    echo "請先執行：python tools/finetune_gemma.py"
    exit 1
fi

mkdir -p "$MERGED_DIR" "$GGUF_DIR"

# ── 步驟 1：合併 LoRA adapters → 完整 float16 模型 ───────────────────────────
echo "=== 步驟 1：合併 LoRA adapters ==="
ROOT="$ROOT" python3 - <<'PYEOF'
from unsloth import FastLanguageModel
from pathlib import Path
import os

root        = Path(os.environ["ROOT"])
adapter_dir = root / "models" / "trafficvision-gemma4"
merged_dir  = root / "models" / "trafficvision-gemma4-merged"

print(f"  從 {adapter_dir} 載入 ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    str(adapter_dir),
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)
print(f"  合併並儲存至 {merged_dir} ...")
model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
print(f"  完成。")
PYEOF

# ── 步驟 2：轉換為 GGUF（Q4_K_M 量化）────────────────────────────────────────
echo ""
echo "=== 步驟 2：轉換為 GGUF（Q4_K_M）==="
if [ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]; then
    echo "ERROR: 找不到 $LLAMA_CPP/convert_hf_to_gguf.py"
    echo "請先執行：git clone https://github.com/ggerganov/llama.cpp.git $LLAMA_CPP"
    exit 1
fi

python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$GGUF_FILE" \
    --outtype q4_k_m

echo "  GGUF 已寫入 $GGUF_FILE"

# ── 步驟 3：建立 Ollama Modelfile ─────────────────────────────────────────────
echo ""
echo "=== 步驟 3：建立 Ollama Modelfile ==="
MODELFILE="$GGUF_DIR/Modelfile"
cat > "$MODELFILE" <<MFEOF
FROM ./${MODEL_TAG}-q4_k_m.gguf

SYSTEM "你是 TrafficVision AI 助理，專門分析台北市北科大周邊路網的即時車流、GRU 預測與號誌優化結果。請使用繁體中文回答，數據需引用具體數值，建議要有依據。"

PARAMETER temperature 0.3
PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.1
MFEOF
echo "  Modelfile 寫入 $MODELFILE"

# ── 步驟 4：匯入 Ollama ───────────────────────────────────────────────────────
echo ""
echo "=== 步驟 4：匯入 Ollama ==="
cd "$GGUF_DIR"
ollama create "$MODEL_TAG" -f Modelfile
echo "  模型已匯入：$MODEL_TAG"

# ── 完成 ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " 完成！"
echo ""
echo " 測試模型："
echo "   ollama run $MODEL_TAG"
echo ""
echo " 啟動 TrafficVision API（自動呼叫此模型）："
echo "   python 'TrafficVision Design System/serve_api.py'"
echo ""
echo " 若要更換模型名稱，設定環境變數："
echo "   OLLAMA_MODEL_NAME=your-model-name python 'TrafficVision Design System/serve_api.py'"
echo "======================================================"
