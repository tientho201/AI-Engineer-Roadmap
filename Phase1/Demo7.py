import numpy as np

vocab  = ["mèo", "chó", "chim", "cá", "ngựa"]
logits = np.array([3.0, 2.5, 1.0, 0.5, 0.2])

def sample(logits, temperature=1.0, top_p=1.0, seed=0):
    rng = np.random.default_rng(seed)
    z = logits / max(temperature, 1e-6)          # temperature thấp -> nhọn hơn -> ít ngẫu nhiên
    """
    z = logits / max(temperature, 1e-6)
    temperature thấp -> nhọn hơn -> ít ngẫu nhiên
    temperature cao -> mềm hơn -> nhiều ngẫu nhiên
    temperature = 0 -> tất định
    temperature = 1 -> mặc định
    temperature = 2 -> rất sáng tạo
    """

    p = np.exp(z - z.max()); p /= p.sum()       # softmax
    
    # Nucleus (top-p): chỉ giữ nhóm token nhỏ nhất có tổng xác suất >= top_p
    idx = np.argsort(-p)    # sắp xếp theo xác suất giảm dần 
    # Chỉ số token được sắp xếp theo xác suất giảm dần.
    # Sắp xếp chỉ số token theo xác suất giảm dần.
    # Dấu -p được dùng vì np.argsort mặc định sắp xếp tăng dần. Sắp xếp -p tăng dần tương đương với sắp xếp p giảm dần.
    
    
    cum = np.cumsum(p[idx]) # tính tổng xác suất tích lũy 
    # Tổng xác suất tích lũy của token được sắp xếp theo xác suất giảm dần.
    
    keep = idx[:np.searchsorted(cum, top_p) + 1] # giữ nhóm token nhỏ nhất có tổng xác suất >= top_p
    # np.searchsorted(cum, top_p) tìm vị trí của top_p trong cum
    
    p_keep = p[keep] / p[keep].sum() # normalize lại xác suất của nhóm token được giữ lại
    return vocab[rng.choice(keep, p=p_keep)], p.round(3)

print("T=0.1 (gần như tất định):", sample(logits, temperature=0.1)[1])
print("T=1.0 (mặc định)        :", sample(logits, temperature=1.0)[1])
print("T=2.0 (rất sáng tạo)    :", sample(logits, temperature=2.0)[1])

