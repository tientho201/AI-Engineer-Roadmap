import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODEL = "Qwen/Qwen3-4B-Instruct"

# 1) Nạp model ở 4-bit -> từ ~16GB xuống ~3GB
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat4 — tốt hơn int4 thường
    bnb_4bit_compute_dtype=torch.bfloat16,  # tính toán ở bf16 cho chính xác
    bnb_4bit_use_double_quant=True,         # lượng tử hoá cả hằng số -> tiết kiệm thêm
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# 2) Gắn adapter LoRA
lora = LoraConfig(
    r=16,                 # rank — càng cao càng nhiều dung lượng học (8/16/32/64)
    lora_alpha=32,        # hệ số scale, thường đặt = 2×r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # gắn vào TẤT CẢ linear
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
# trainable: 32M || all: 4B || trainable%: 0.8   <- chỉ train 0.8% tham số!

# 3) Chuẩn bị dữ liệu đúng định dạng chat
def to_chat(ex):
    return {"text": tokenizer.apply_chat_template([
        {"role": "user", "content": ex["instruction"]},
        {"role": "assistant", "content": ex["output"]},
    ], tokenize=False)}

ds = load_dataset("json", data_files="train.jsonl")["train"].map(to_chat)

# 4) Train
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=SFTConfig(
        output_dir="./qlora-out",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,      # batch hiệu dụng = 16
        learning_rate=2e-4,                 # LoRA dùng lr CAO hơn full FT nhiều
        lr_scheduler_type="cosine",
        warmup_steps=0.03,   # float trong [0,1) = tỉ lệ warmup (thay cho warmup_ratio cũ)
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=2048,     
        gradient_checkpointing=True,    # dùng checkpointing để tiết kiệm VRAM
    ),
)
trainer.train()
model.save_pretrained("./qlora-adapter")     # chỉ ~50MB, không phải 8GB