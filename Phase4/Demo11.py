from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()


@dataclass
class Chunk:
    text: str
    source: str
    page: int

RAG_PROMPT = """Bạn là trợ lý trả lời dựa trên tài liệu.

QUY TẮC:
1. CHỈ dùng thông tin trong <context>. Không dùng kiến thức riêng.
2. Mọi khẳng định phải kèm trích dẫn đúng nhãn trong context, dạng [tên_file:số_trang],
   ví dụ [handbook.pdf:12]. Không thêm chữ khác vào trong ngoặc.
3. Nếu context KHÔNG đủ thông tin, trả lời chính xác:
   "Tôi không tìm thấy thông tin này trong tài liệu."
   Tuyệt đối không bịa.
4. Trả lời ngắn gọn, đúng trọng tâm.

<context>
{context}
</context>

Câu hỏi: {question}"""


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{c.source}:{c.page}]\n{c.text}" for c in chunks)


def answer(question: str, chunks: list[Chunk]) -> str:
    client =  OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful assistant that answers questions based on the provided context.",
        max_output_tokens=1024,
        temperature=0,
        input=RAG_PROMPT.format(
            context=build_context(chunks), question=question),
    )
    return resp.output_text


chunks = [
    Chunk("Nhân viên chính thức được 12 ngày phép năm.", "handbook.pdf", 12),
    Chunk("Phép năm không dùng hết được chuyển tối đa 5 ngày sang năm sau.",
          "handbook.pdf", 13),
]
print(answer("Tôi được bao nhiêu ngày phép?", chunks))