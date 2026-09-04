from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

docs = [
    "Cấu hình timeout cho API gữi request bằng biến REQUEST_TIMEOUT_MS",
    "Hướng dẫn tăng thời gian chờ phản hồi của dịch vụ",
    "Cách nấu mì tôm ngon",
    "Lỗi ERR_CONN_2048 xuất hiện khi kết nối quá tải",
]

# --- BM25: tìm từ khoá CHÍNH XÁC ---
bm25 = BM25Okapi([d.lower().split() for d in docs]) # tạo BM25Okapi với các tài liệu đã tokenize

# --- Embedding: tìm theo Ý NGHĨA ---
model = SentenceTransformer("intfloat/multilingual-e5-base")
doc_emb = model.encode(docs, normalize_embeddings=True)

for query in ["ERR_CONN_2048", "làm sao để chờ lâu hơn"]:
    bm = bm25.get_scores(query.lower().split())
    vec = model.encode([query], normalize_embeddings=True) @ doc_emb.T
    print(f"\nQuery: {query}")
    print(f"  BM25 thắng     -> {docs[bm.argmax()][:50]}")
    print(f"  Embedding thắng -> {docs[vec[0].argmax()][:50]}")

# KẾT LUẬN: BM25 giỏi mã lỗi, tên riêng, số hiệu.
#            Embedding giỏi câu hỏi diễn đạt khác đi.
#            Production PHẢI dùng cả hai (hybrid search).