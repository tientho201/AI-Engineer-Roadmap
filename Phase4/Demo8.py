import numpy as np , re 
from sentence_transformers import SentenceTransformer

def semantic_chunk(text: str, model, threshold_pct: int = 25, min_chars: int = 200, max_chars: int = 1500) -> list[str]:
    """Cắt ở nơi Ý NGHĨA THAY ĐỔI, không phải ở số ký tự cố định."""
    sens = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text) if s.strip()]
    if  len(sens) < 3:
        return [text]
    
    emb = model.encode(sens , normalize_embeddings=True)
    
    # tính toán độ tương tự cosine giữa các embedding
    sims = np.array([np.dot(emb[i], emb[i+1]) for i in range(len(sens) - 1)])
    
    # Ngưỡng tương tự để xác định chỗ cắt
    cut_threshold = np.percentile(sims, threshold_pct)   
    chunks, cur = [] , sens[0]
    for i , s in enumerate(sens[1:]):
        too_long = len(cur) + len(s) > max_chars
        topic_shift = sims[i] < cut_threshold and len(cur) >= min_chars
        if too_long or topic_shift:
            chunks.append(cur)
            cur = s
        else:
            cur += " " + s
    chunks.append(cur)
    return chunks
model = SentenceTransformer("intfloat/multilingual-e5-base")
text = """Python là ngôn ngữ thông dịch bậc cao. Nó được Guido van Rossum tạo ra năm 1991.
Python nổi tiếng nhờ cú pháp đơn giản và dễ đọc.
Trong khi đó, ẩm thực Việt Nam rất đa dạng vùng miền. Phở là món ăn đại diện của miền Bắc.
Bún bò Huế mang đậm hương vị miền Trung."""

for i, c in enumerate(semantic_chunk(text, model, min_chars=50)):
    print(f"--- Chunk {i} ---\n{c}\n")