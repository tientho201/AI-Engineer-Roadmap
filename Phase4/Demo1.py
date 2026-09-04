from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")

texts = [
    "Hello world",
    "Xin chào thế giới",                   # tiếng Việt tốn NHIỀU token hơn
    "antidisestablishmentarianism",
    "def calculate_total(items): return sum(items)",
]
for t in texts:
    ids = tok.encode(t)
    print(f"{len(ids):3d} token | {t}")
    print(f"          {tok.convert_ids_to_tokens(ids)}\n")

# BÀI HỌC KINH TẾ:
# Tiếng Việt thường tốn gấp 2-3 lần token so với tiếng Anh cùng nội dung
# -> chi phí API cao hơn, context window hiệu dụng nhỏ hơn.
# Luôn đo token thật trước khi ước tính chi phí dự án.