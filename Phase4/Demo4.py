from pydantic import BaseModel, Field
from typing import Literal
import instructor
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()   
# 1) Định nghĩa schema — đây là hợp đồng giữa LLM và code của bạn
class TicketAnalysis(BaseModel):
    sentiment: Literal["tích cực", "trung tính", "tiêu cực"]
    urgency: int = Field(ge=1, le=5, description="1=thấp, 5=khẩn cấp")
    category: Literal["thanh toán", "kỹ thuật", "tài khoản", "khác"]
    summary: str = Field(max_length=200)
    needs_human: bool
    keywords: list[str] = Field(max_length=5)


client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

result = client.messages.create(
    model="gpt-4o-mini",
    max_tokens=1024,
    response_model=TicketAnalysis,      # <- tự validate + retry nếu sai schema
    messages=[{
        "role": "user",
        "content": "Phân tích ticket: 'Tôi đã bị trừ tiền 3 lần cho cùng một đơn "
                   "hàng! Gọi hotline không ai bắt máy. Quá tệ!'"
    }],
)

print(result.model_dump_json(indent=2, ensure_ascii=False))
print(type(result.urgency))   # <class 'int'> — đã là object Python, không phải string