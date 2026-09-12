from pydantic import BaseModel
from typing import Literal
import instructor
from anthropic import Anthropic

class Complexity(BaseModel):
    level: Literal["simple", "medium", "complex"]
    reason: str

ROUTE = {
    "simple":  "claude-haiku-4-5",     # FAQ, phân loại, trích xuất
    "medium":  "claude-sonnet-4-6",    # phân tích, viết lách
    "complex": "claude-opus-5",        # suy luận nhiều bước, kiến trúc
}

client = instructor.from_anthropic(Anthropic())
cheap = Anthropic()


def smart_route(query: str) -> str:
    # Dùng model RẺ NHẤT để phân loại -> chi phí routing không đáng kể
    c = client.messages.create(
        model="claude-haiku-4-5", max_tokens=256,
        response_model=Complexity,
        messages=[{"role": "user", "content":
            f"Phân loại độ khó của yêu cầu sau:\n{query}"}])
    return ROUTE[c.level]


# CHIẾN LƯỢC GIẢM CHI PHÍ — xếp theo hiệu quả:
# 1. Prompt caching       : -50% đến -90% (nếu system prompt dài)
# 2. Model routing        : -40% đến -70%
# 3. Semantic cache       : -20% đến -40% (nếu câu hỏi lặp nhiều)
# 4. Rút gọn prompt       : -10% đến -30%
# 5. Giới hạn max_tokens  : -5% đến -20%
# 6. Tự host model nhỏ     : -60%+ nhưng tăng chi phí vận hành