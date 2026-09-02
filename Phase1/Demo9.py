from dataclasses import dataclass
from functools import lru_cache
import time, asyncio

# 1) DATACLASS -> cấu hình model (thay cho dict lộn xộn)
@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 32
    epochs: int = 10
    model_name: str = "bert-base"

cfg = TrainConfig(lr=1e-4)


# 2) GENERATOR -> đọc file 100GB mà chỉ tốn vài MB RAM
def read_batches(path, batch_size=32):
    batch = []
    with open(path) as f:
        for line in f:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield batch          # trả về rồi TẠM DỪNG, không giữ toàn bộ file
                batch = []
    if batch:
        yield batch

# 3) DECORATOR -> đo thời gian, retry, cache
def timeit(fn):
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        r = fn(*a, **kw)
        print(f"[{fn.__name__}] {time.perf_counter()-t0:.4f}s")
        return r
    return wrapper

"""
def my_decorator(fn):
    def wrapper(*args, **kwargs):
        # làm gì đó trước khi gọi hàm gốc
        result = fn(*args, **kwargs)
        # làm gì đó sau khi gọi hàm gốc
        return result
    return wrapper
"""

@timeit
@lru_cache(maxsize=128)          # cache kết quả -> gọi lại tức thì
def embed(text: str):
    time.sleep(0.3)              # giả lập gọi API embedding
    return [hash(text) % 100 / 100] * 4

embed("xin chào")   # ~0.3s
embed("xin chào")   # ~0.0s (cache hit)

# 4) ASYNC -> gọi 10 request LLM song song thay vì tuần tự
async def call_llm(prompt: str, i: int):
    await asyncio.sleep(1)       # giả lập độ trễ mạng
    return f"trả lời {i}"

async def main():
    t0 = time.perf_counter()
    results = await asyncio.gather(*[call_llm("hi", i) for i in range(10)])
    print(results)
    print(f"10 request trong {time.perf_counter()-t0:.2f}s")  # ~1s thay vì ~10s

asyncio.run(main())

"""
*[...] nghĩa là: lấy các phần tử trong list và truyền từng phần tử như một đối số riêng biệt.
asyncio.gather(*[call_llm("hi", i) for i in range(10)])

Vì asyncio.gather() nhận nhiều coroutine theo dạng: asyncio.gather(task1, task2, task3)
"""