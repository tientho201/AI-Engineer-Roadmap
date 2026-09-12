"""
Vấn đề: chunk bị tách khỏi ngữ cảnh.
  Chunk gốc: "Doanh thu tăng 3% so với quý trước."
  -> Doanh thu của công ty nào? Quý nào? Không ai biết.

Giải pháp: dùng LLM thêm 1-2 câu ngữ cảnh vào ĐẦU mỗi chunk TRƯỚC khi embed.
Dùng prompt caching cho toàn văn bản -> chi phí rất rẻ.
"""
from anthropic import Anthropic

client = Anthropic()

CTX_PROMPT = """<document>{doc}</document>

Đây là một đoạn trích từ tài liệu trên:
<chunk>{chunk}</chunk>

Viết 1-2 câu ngắn đặt đoạn này vào ngữ cảnh của toàn tài liệu,
nhằm giúp tìm kiếm tốt hơn. Chỉ trả về câu ngữ cảnh, không gì khác."""


def contextualize(doc: str, chunks: list[str]) -> list[str]:
    out = []
    for chunk in chunks:
        resp = client.messages.create(
            model="claude-haiku-4-5", max_tokens=200,
            system=[{"type": "text", "text": f"<document>{doc}</document>",
                     "cache_control": {"type": "ephemeral"}}],  # <- CACHE
            messages=[{"role": "user", "content":
                       CTX_PROMPT.format(doc="(xem trên)", chunk=chunk)}])
        out.append(resp.content[0].text.strip() + "\n\n" + chunk)
    return out

# Kết hợp contextual embedding + contextual BM25 + rerank
# thường giảm mạnh tỉ lệ truy xuất thất bại so với RAG cơ bản.