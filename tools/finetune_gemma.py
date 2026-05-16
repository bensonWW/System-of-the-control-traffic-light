#!/usr/bin/env python3
"""
使用 Unsloth 對 Gemma 4 4B 進行 LoRA 指令微調（8-bit 量化，適合 8GB VRAM）。

安裝依賴：
    pip install -r requirements-training.txt

執行前先生成資料集：
    python tools/generate_finetune_dataset.py

執行方式（本機 NVIDIA GPU）：
    python tools/finetune_gemma.py

輸出：
    models/trafficvision-gemma4/  ← LoRA adapters + tokenizer

下一步：
    bash tools/export_to_gguf.sh  ← 合併、量化、匯入 Ollama
"""
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATASET    = ROOT / "data" / "finetune_dataset.jsonl"
OUTPUT_DIR = ROOT / "models" / "trafficvision-gemma4"

# Gemma 4 E4B（4-bit NF4）：適合 8GB VRAM（RTX 4060 Ti）
# 4-bit E4B 模型只占 ~2.5GB 權重，Unsloth 針對 4-bit 深度優化。
# 如要換回 12B 請確保有 16GB+ VRAM，並將 MODEL_NAME 改為：
#   unsloth/gemma-4-12b-it-unsloth-bnb-4bit
MODEL_NAME  = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
MAX_SEQ_LEN = 1024   # 135 筆短問答，1024 已足夠，省 VRAM
LORA_RANK   = 16
BATCH_SIZE  = 1
GRAD_ACCUM  = 8      # effective batch = 1×8 = 8（同原始設定）
EPOCHS      = 3
LR          = 2e-4

# ── 載入模型（4-bit NF4 量化）─────────────────────────────────────────────────
print(f"載入 {MODEL_NAME}（4-bit）...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = MODEL_NAME,
    max_seq_length= MAX_SEQ_LEN,
    dtype         = None,
    load_in_4bit  = True,
    device_map    = {"": 0},   # 強制所有 layer 放 GPU 0，避免 CPU offload 衝突
)

model = FastLanguageModel.get_peft_model(
    model,
    r                  = LORA_RANK,
    target_modules     = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha         = LORA_RANK,
    lora_dropout       = 0,
    bias               = "none",
    use_gradient_checkpointing = "unsloth",  # 降低 VRAM 用量
    random_state       = 42,
)

# ── 載入並格式化資料集 ────────────────────────────────────────────────────────
print(f"載入資料集 {DATASET} ...")
if not DATASET.exists():
    raise FileNotFoundError(
        f"找不到 {DATASET}，請先執行：python tools/generate_finetune_dataset.py"
    )

dataset = load_dataset("json", data_files=str(DATASET), split="train")

def format_chat(example):
    """conversations 格式 → Gemma chat template 單一 text 欄位。"""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = [
        {"role": role_map.get(t["from"], t["from"]), "content": t["value"]}
        for t in example["conversations"]
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }

dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
print(f"  {len(dataset)} 筆訓練樣本")

# ── 訓練 ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        output_dir                  = str(OUTPUT_DIR),
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        save_strategy               = "epoch",
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        seed                        = 42,
        report_to                   = "none",
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LEN,
        packing                     = True,
    ),
)

print("開始訓練（Ctrl+C 可中斷並保留最新 checkpoint）...")
trainer.train()

# ── 儲存 LoRA adapters ────────────────────────────────────────────────────────
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"\nLoRA adapters 已儲存 → {OUTPUT_DIR}")
print("下一步：bash tools/export_to_gguf.sh")
