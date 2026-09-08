from anthropic import Anthropic

client = Anthropic()
LONG_CONTEXT = open("company_handbook.txt").read()   # ~50.000 token

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {"type": "text", "text": "Bạn là trợ lý nội bộ của công ty."},
        {
            "type": "text",
            "text": LONG_CONTEXT,
            "cache_control": {"type": "ephemeral"},   # <- ĐÁNH DẤU CACHE TẠI ĐÂY
        },
    ],
    messages=[{"role": "user", "content": "Chính sách nghỉ phép như thế nào?"}],
)

u = resp.usage
print(f"Cache ghi mới : {u.cache_creation_input_tokens}")
print(f"Cache đọc lại : {u.cache_read_input_tokens}")   # lần 2 trở đi sẽ rất lớn
print(f"Input thường  : {u.input_tokens}")

# QUY TẮC: đặt phần TĨNH (system prompt, tài liệu, few-shot) ở ĐẦU và cache,
# phần ĐỘNG (câu hỏi người dùng) ở CUỐI.