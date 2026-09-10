from langfuse import observe, get_client, propagate_attributes
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
langfuse = get_client()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@observe(name="retrieve")
def retrieve(query: str) -> list[str]:
    return ["chunk 1", "chunk 2"]


@observe(name="generate", as_type="generation")
def generate(query: str, chunks: list[str]) -> str:
    model_name = "gpt-4o-mini"
    resp = client.chat.completions.create(
        model=model_name, max_tokens=512,
        messages=[{"role": "user",
                   "content": f"Context: {chunks}\n\nCâu hỏi: {query}"}])

    # @observe không tự đọc được model/usage từ response OpenAI,
    # phải set thủ công thì Langfuse mới tính được cost.
    langfuse.update_current_generation(
        model=model_name,
        usage_details={
            "input": resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
        },  # "total" không bắt buộc, Langfuse backend tự cộng
    )
    return resp.choices[0].message.content


@observe(name="rag_pipeline")
def rag(query: str, user_id: str) -> str:
    # propagate_attributes: gắn user_id/tags cho span hiện tại VÀ mọi span con
    # (retrieve, generate) được tạo bên trong khối with này.
    with propagate_attributes(user_id=user_id, tags=["prod", "rag-v2"]):
        chunks = retrieve(query)
        return generate(query, chunks)


answer = rag("Chính sách nghỉ phép?", user_id="user-123")

# Dashboard sẽ cho bạn thấy: từng bước, độ trễ, số token, chi phí, điểm eval
# Metric vận hành cần theo dõi:
#   - TTFT (độ trễ cảm nhận)
#   - chi phí token / phiên (phát hiện prompt phình to)
#   - cache hit rate
#   - số bước agent (phát hiện agent lặp vô hạn đốt token)