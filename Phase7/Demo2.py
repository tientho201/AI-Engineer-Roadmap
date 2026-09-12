from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Calibration set: PHẢI giống phân phối dữ liệu thực tế bạn sẽ dùng.
# Dùng calibration sai là nguyên nhân số 1 khiến model quantize bị tệ.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")

oneshot(
    model=model,
    dataset=ds,
    recipe=GPTQModifier(targets="Linear", scheme="W4A16",
                        ignore=["lm_head"]),   # KHÔNG quantize lm_head
    max_seq_length=2048,
    num_calibration_samples=512,
    output_dir="./Qwen3-8B-W4A16",
)

# Sau khi quantize, BẮT BUỘC đo lại chất lượng:
# lm_eval --model vllm --model_args pretrained=./Qwen3-8B-W4A16 \
#         --tasks gsm8k,mmlu --batch_size auto