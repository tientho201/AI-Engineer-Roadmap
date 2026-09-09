from sentence_transformers import CrossEncoder
from Demo9 import HybridRetriever 

docs = [
    "Lỗi ERR_CONN_2048 xảy ra khi số kết nối đồng thời vượt 2048.",
    "Để tăng thời gian chờ, chỉnh biến môi trường REQUEST_TIMEOUT_MS.",
    "Hướng dẫn cấu hình connection pool cho PostgreSQL.",
    "Cách khắc phục tình trạng quá tải kết nối mạng.",
]
r = HybridRetriever(docs)
# Bi-encoder (embedding): mã hoá query và doc RIÊNG -> nhanh, kém chính xác
# Cross-encoder (rerank): đọc query VÀ doc CÙNG LÚC -> chậm, chính xác hơn nhiều
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

query = "Làm sao tăng thời gian chờ phản hồi?"
candidates = [d for d, _ in r.search(query, top_k=20)]   # lấy 20 từ hybrid

scores = reranker.predict([(query, c) for c in candidates]) # đọc query VÀ doc CÙNG LÚC => list of scores
ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:3]  # giữ 3

for doc, s in ranked:
    print(f"{s:+.3f} | {doc}")

# CÔNG THỨC SẢN XUẤT: retrieve 20 -> rerank còn 5 -> đưa 3-5 vào LLM.
# Rerank trên 100 ứng viên hầu như không có thêm lợi ích.