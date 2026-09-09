import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self, docs: list[str], model_name="intfloat/multilingual-e5-base"):
        self.docs = docs
        self.model = SentenceTransformer(model_name , device="cpu")
        self.emb = self.model.encode([f"passage: {d}" for d in docs],
                                     normalize_embeddings=True)
        self.bm25 = BM25Okapi([d.lower().split() for d in docs])

    def search(self, query: str, top_k: int = 20, k_rrf: int = 60):
        """RRF: score = Σ 1/(k + rank). Ưu điểm lớn nhất là không cần
        chuẩn hoá thang điểm giữa BM25 và cosine — chỉ dùng THỨ HẠNG."""
        bm_rank  = np.argsort(-self.bm25.get_scores(query.lower().split()))
        vec_rank = np.argsort(-(self.emb @ self.model.encode(
            f"query: {query}", normalize_embeddings=True)))

        scores = {}
        for rank, idx in enumerate(bm_rank):
            scores[idx] = scores.get(idx, 0) + 1 / (k_rrf + rank + 1)
        for rank, idx in enumerate(vec_rank):
            scores[idx] = scores.get(idx, 0) + 1 / (k_rrf + rank + 1)

        best = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(self.docs[i], round(s, 5)) for i, s in best]


docs = [
    "Lỗi ERR_CONN_2048 xảy ra khi số kết nối đồng thời vượt 2048.",
    "Để tăng thời gian chờ, chỉnh biến môi trường REQUEST_TIMEOUT_MS.",
    "Hướng dẫn cấu hình connection pool cho PostgreSQL.",
    "Cách khắc phục tình trạng quá tải kết nối mạng.",
]
r = HybridRetriever(docs)
for doc, s in r.search("ERR_CONN_2048 là lỗi gì", top_k=3):
    print(f"{s} | {doc}")