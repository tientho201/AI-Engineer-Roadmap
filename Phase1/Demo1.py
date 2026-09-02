# Vector & Cosine Similarity (nền tảng của mọi vector database)
import numpy as np
# np.linalg.norm:  tính chuẩn (norm) của một vectơ hoặc một ma trận
def consine_similarity(a: np.ndarray , b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Giả lập embedding 5 chiều của 3 câu
emb = {
    "con mèo ngồi trên thảm": np.array([0.9, 0.1, 0.8, 0.2, 0.1]),
    "con mèo nằm trên chiếu": np.array([0.85, 0.15, 0.75, 0.25, 0.05]),
    "lãi suất ngân hàng tăng": np.array([0.1, 0.9, 0.05, 0.8, 0.7]),
}

query = emb["con mèo ngồi trên thảm"]
for text, vec in emb.items(): 
    similarity = consine_similarity(query, vec)
    print(f"Câu {text} có độ tương đồng: {similarity}")
    
# Output:
# Câu con mèo ngồi trên thảm có độ tương đồng: 1.0000000000000002
# Câu con mèo nằm trên chiếu có độ tương đồng: 0.9967982132837949
# Câu lãi suất ngân hàng tăng có độ tương đồng: 0.2620767647263377