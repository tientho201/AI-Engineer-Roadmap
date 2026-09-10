from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from Demo16 import model, tokenizer

# DPO cần dữ liệu dạng CẶP: cùng prompt, 1 câu trả lời được chọn / 1 bị loại
data = Dataset.from_dict({
    "prompt":   ["Giải thích RAG là gì?"],
    "chosen":   ["RAG là kỹ thuật kết hợp truy xuất tài liệu với sinh văn bản..."],
    "rejected": ["RAG là một thứ gì đó liên quan đến AI."],
})

trainer = DPOTrainer(
    model=model,                # model đã SFT trước đó
    ref_model=None,             # None -> dùng bản đóng băng của chính model
    train_dataset=data,
    processing_class=tokenizer,
    args=DPOConfig(
        output_dir="./dpo-out",
        beta=0.1,               # càng cao càng bám sát model gốc
        learning_rate=5e-7,     # DPO dùng lr RẤT nhỏ
        num_train_epochs=1,
        per_device_train_batch_size=2,
        bf16=True,
    ),
)
trainer.train()

# DPO vs RLHF: DPO bỏ hoàn toàn reward model và vòng PPO
# -> đơn giản hơn nhiều, ổn định hơn, kết quả tương đương.
# Năm 2026 hầu hết team dùng DPO thay RLHF.