#!/usr/bin/env python3
"""
export_to_gguf.py — Windows 版 GGUF 匯出 + Ollama Modelfile 生成

執行方式：
    python tools/export_to_gguf.py

前置條件：
    - 已完成 python tools/finetune_gemma.py（models/trafficvision-gemma4/ 存在）
    - Ollama 已安裝：https://ollama.com/download

輸出：
    models/trafficvision-gemma4-merged/   （合併後 fp16 模型，可刪除）
    models/gguf/trafficvision-gemma4-q8_0.gguf
    models/gguf/Modelfile
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT        = Path(__file__).parent.parent
ADAPTER_DIR = ROOT / "models" / "trafficvision-gemma4"
MERGED_DIR  = ROOT / "models" / "trafficvision-gemma4-merged"
GGUF_DIR    = ROOT / "models" / "gguf"
MODEL_TAG   = os.environ.get("OLLAMA_MODEL_NAME", "trafficvision-gemma4")
GGUF_FILE   = GGUF_DIR / f"{MODEL_TAG}-q8_0.gguf"
MODELFILE   = GGUF_DIR / "Modelfile"

# convert_hf_to_gguf.py is shipped with llama.cpp (cloned by Unsloth)
LLAMA_CPP_DIR    = Path.home() / ".unsloth" / "llama.cpp"
CONVERT_SCRIPT   = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"


def check_prerequisites():
    if not ADAPTER_DIR.exists() or not (ADAPTER_DIR / "adapter_model.safetensors").exists():
        sys.exit(f"ERROR: 找不到 {ADAPTER_DIR}，請先執行 python tools/finetune_gemma.py")
    if not CONVERT_SCRIPT.exists():
        sys.exit(
            f"ERROR: 找不到 {CONVERT_SCRIPT}\n"
            "請確認 Unsloth 已下載 llama.cpp，或手動執行：\n"
            f"  git clone https://github.com/ggerganov/llama.cpp {LLAMA_CPP_DIR}"
        )


def merge_lora():
    print("=== 步驟 1/3：合併 LoRA adapters → fp16 模型 ===")
    print(f"  來源：{ADAPTER_DIR}")
    print(f"  目標：{MERGED_DIR}")

    if MERGED_DIR.exists() and (MERGED_DIR / "config.json").exists():
        print("  合併模型已存在，跳過合併步驟。")
        return

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    from unsloth import FastLanguageModel

    print("  載入 LoRA 模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = str(ADAPTER_DIR),
        max_seq_length= 1024,
        dtype         = None,
        load_in_4bit  = True,
        device_map    = {"": 0},
    )

    print("  合併 LoRA 並以 fp16 儲存（需數分鐘）...")
    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"  合併完成：{MERGED_DIR}")


def convert_to_gguf():
    print("\n=== 步驟 2/3：轉換為 GGUF（Q8_0）===")
    print(f"  來源：{MERGED_DIR}")
    print(f"  目標：{GGUF_FILE}")

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(MERGED_DIR),
        "--outtype", "q8_0",
        "--outfile", str(GGUF_FILE),
    ]
    print(f"  執行：{' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, text=True)
    print(f"  GGUF 已儲存：{GGUF_FILE}")


def write_modelfile():
    print("\n=== 步驟 3/3：建立 Ollama Modelfile ===")
    content = f"""FROM ./{GGUF_FILE.name}

SYSTEM "你是 TrafficVision AI 助理，專門分析台北市北科大周邊路網的即時車流、GRU 預測與號誌優化結果。請使用繁體中文回答，數據需引用具體數值，建議要有依據，避免含糊描述。"

PARAMETER temperature 0.3
PARAMETER num_ctx 16384
PARAMETER repeat_penalty 1.1
"""
    MODELFILE.write_text(content, encoding="utf-8")
    print(f"  Modelfile 已寫入：{MODELFILE}")

    print("\n  匯入 Ollama...")
    try:
        result = subprocess.run(
            ["ollama", "create", MODEL_TAG, "-f", str(MODELFILE)],
            cwd=str(GGUF_DIR),
            check=True,
            text=True,
            capture_output=True,
        )
        print(result.stdout)
        print(f"  模型已匯入：{MODEL_TAG}")
        print(f"  測試：ollama run {MODEL_TAG}")
    except FileNotFoundError:
        print("  找不到 ollama 指令。請先安裝 Ollama：https://ollama.com/download")
        print(f"\n  安裝完成後，執行以下指令手動匯入：")
        print(f"    cd \"{GGUF_DIR}\"")
        print(f"    ollama create {MODEL_TAG} -f Modelfile")
    except subprocess.CalledProcessError as e:
        print(f"  ollama create 失敗：{e.stderr}")
        print(f"  請確認 Ollama 服務正在執行：ollama serve")


if __name__ == "__main__":
    print("=" * 56)
    print(" TrafficVision Gemma 4 E4B — GGUF Export & Ollama Import")
    print("=" * 56)
    check_prerequisites()
    merge_lora()
    convert_to_gguf()
    write_modelfile()
    print("\n完成！")
    print(f"GGUF 路徑：{GGUF_FILE}")
    print(f"Modelfile ：{MODELFILE}")
    print(f"\n合併後的 fp16 模型可刪除以釋放空間：")
    print(f"  rmdir /s /q \"{MERGED_DIR}\"")
