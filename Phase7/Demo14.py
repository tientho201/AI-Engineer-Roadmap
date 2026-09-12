import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
import time

@dataclass
class SemanticCache:
    model: SentenceTransformer
    threshold: float = 0.93
    ttl_seconds: int = 3600
    _keys: list = field(default_factory=list)
    _vals: list = field(default_factory=list)
    _emb: list = field(default_factory=list)
    _ts: list = field(default_factory=list)

    def get(self, query: str):
        if not self._emb:
            return None
        now = time.time()
        q = self.model.encode(query, normalize_embeddings=True)
        sims = np.array(self._emb) @ q
        i = int(sims.argmax())
        if sims[i] >= self.threshold and now - self._ts[i] < self.ttl_seconds:
            return self._vals[i]
        return None

    def set(self, query: str, value: str):
        self._keys.append(query)
        self._vals.append(value)
        self._emb.append(self.model.encode(query, normalize_embeddings=True))
        self._ts.append(time.time())


cache = SemanticCache(SentenceTransformer("intfloat/multilingual-e5-base"))
cache.set("Chính sách nghỉ phép như thế nào?", "Bạn được 12 ngày phép năm.")

print(cache.get("Tôi được nghỉ phép bao nhiêu ngày?"))   # cache hit
print(cache.get("Giờ làm việc là mấy giờ?"))              # None

# CẠM BẪY: ngưỡng quá thấp -> trả sai câu trả lời cho câu hỏi khác.
# Luôn bắt đầu từ ngưỡng CAO (0.95+) rồi hạ dần, đo tỉ lệ cache sai.
# KHÔNG dùng semantic cache cho câu hỏi có yếu tố cá nhân hoặc thời gian.