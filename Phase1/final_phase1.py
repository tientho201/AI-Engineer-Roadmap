from dataclasses import dataclass , field
import numpy as np
import re 

@dataclass
class MiniVectorDB:
    dim: int = 64
    _vocab: dict[str, int] = field(default_factory=dict)
    _idf: np.ndarray | None = None
    _docs: list = field(default_factory=list)
    _matrix: np.ndarray | None = None


    def _tokenize(self, doc: str) -> list[str]:
        return re.findall(r'\w+', doc.lower()) # Tìm tất cả các từ trong doc và trả về danh sách các từ.

    def _fit_tfidf(self) -> None:
        self._vocab = {}
        doc_tokens = [self._tokenize(doc) for doc in self._docs]
        # Tạo từ điển từ tất cả các từ trong tất cả các docs.
        for tokens in doc_tokens:
            for tok in tokens:
                if tok not in self._vocab:
                    self._vocab[tok] = len(self._vocab)

        n_docs = len(self._docs)
        df = np.zeros(len(self._vocab)) # document frequency: số lượng docs chứa từ đó. Tạo mảng tạo số 0 với số lượng từ trong từ điển.
        for tokens in doc_tokens:
            for tok in set(tokens):
                df[self._vocab[tok]] += 1.0 # tăng 1 cho từ đó.

        # Smooth IDF để tránh chia cho 0 và giữ trọng số ổn định với corpus nhỏ.
        self._idf = np.log((1 + n_docs) / (1 + df)) + 1
        #IDF là inverse document frequency.
        #Ý nghĩa: từ càng hiếm trong corpus thì điểm càng cao.
        
        self._matrix = np.vstack([self._embed(doc) for doc in self._docs])
    
    def _embed(self, doc: str) -> np.ndarray:
        v = np.zeros(len(self._vocab))
        tokens = self._tokenize(doc)
        for tok in tokens:
            if tok in self._vocab:
                v[self._vocab[tok]] += 1.0

        if tokens:
            v = v / len(tokens)
        if self._idf is not None:
            v = v * self._idf

        n = np.linalg.norm(v) # dùng Euclidean norm, còn gọi là L2 norm.
        return v / n if n > 0 else v # Normalize the vector v.
    
    def add(self, docs: list[str]) -> None:
        self._docs.extend(docs)
        self._fit_tfidf()

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, str]]:
        if self._matrix is None:
            return []
        q = self._embed(query)
        scores = self._matrix @ q                 # 1 phép nhân ma trận = so sánh với TẤT CẢ docs
        idx = np.argsort(-scores)[:top_k] # sắp xếp theo điểm số giảm dần và lấy top_k kết quả. Lấy theo index.
        """
        Ví dụ:
            scores = np.array([0.13, 0.18, 0.28, 0.13])
            np.argsort(scores)
        kết quả:
            [0, 3, 1, 2]
        """
        return [(float(scores[i]), self._docs[i]) for i in idx]


    def search_batch(self, queries: list[str], top_k: int = 3) -> list[list[tuple[float, str]]]:
        if self._matrix is None:
            return []
        queries_matrix = np.vstack([self._embed(q) for q in queries])
        scores = self._matrix @ queries_matrix.T # ma trận điểm: mỗi dòng là doc, mỗi cột là query
        results = []
        for query_idx in range(len(queries)):
            query_scores = scores[:, query_idx]
            idx = np.argsort(-query_scores)[:top_k]
            results.append([(float(query_scores[i]), self._docs[i]) for i in idx])
        return results

if __name__ == "__main__":
    db = MiniVectorDB(dim=128)
    db.add([
        "Python là ngôn ngữ lập trình phổ biến cho machine learning",
        "PyTorch là thư viện deep learning của Meta",
        "Cách nấu phở bò truyền thống Hà Nội",
        "Transformer sử dụng cơ chế self-attention",
        "Bún chả là món ăn nổi tiếng của Hà Nội",
    ])

    for score, doc in db.search("thư viện deep learning", top_k=3):
        print(f"{score:.4f} | {doc}")
        
    print("--------------------------------")
    queries = ["thư viện deep learning", "Python là ngôn ngữ lập trình phổ biến cho machine learning"]
    results = db.search_batch(queries, top_k=3)
    for query, result in zip(queries, results):
        print(f"Query: {query}")
        for score, doc in result:
            print(f"{score:.4f} | {doc}")
        print("--------------------------------")
