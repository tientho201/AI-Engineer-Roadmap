from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LONG_CONTEXT = open("company_handbook.txt").read()   # ~50.000 token

# OpenAI KHÔNG dùng prompt=[{...}] như Anthropic system[].
# - prompt  = object template {id, variables} trên dashboard (không phải nội dung)
# - Cache   = tự động theo PREFIX ổn định + prompt_cache_key
# Đặt phần TĨNH (instructions + handbook) ở ĐẦU, câu hỏi ĐỘNG ở CUỐI (input).

resp = client.responses.create(
    model="gpt-4o-mini",
    max_output_tokens=1024,
    instructions=(
        "Bạn là trợ lý nội bộ của công ty. Trả lời dựa trên sổ tay dưới đây.\n\n"
        + LONG_CONTEXT
    ),
    input="Chính sách nghỉ phép như thế nào?",
    prompt_cache_key="techvina-handbook-v1",  # cùng key → dễ cache hit hơn
)

u = resp.usage
cached = getattr(u.input_tokens_details, "cached_tokens", None) if u.input_tokens_details else 0
print(f"Input tokens  : {u.input_tokens}")
print(f"Cached tokens : {cached}")          # lần 2 trở đi (cùng prefix) sẽ tăng
print(f"Output tokens : {u.output_tokens}")
print(resp.output_text)

"""
Lần 1: Cached_tokens: 0 
Lần 2: Cached_tokens: 45824 => Input_tokens: 45988 
""" 

# QUY TẮC: đặt phần TĨNH (system/instructions, tài liệu) ở ĐẦU,
# phần ĐỘNG (câu hỏi) ở CUỐI — giống Anthropic, khác cách đánh dấu cache.

