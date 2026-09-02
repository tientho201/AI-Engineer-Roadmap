import torch, math

# --- Cách cũ: sinusoidal (paper 2017) ---
def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div) # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    pe[:, 1::2] = torch.cos(pos * div) # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    return pe

# --- Cách hiện đại: RoPE (LLaMA, Qwen, Mistral đều dùng) ---
def rope(x, base=10000):
    """Xoay vector Q/K theo vị trí -> tích vô hướng chỉ phụ thuộc KHOẢNG CÁCH tương đối.
    Đây là lý do các LLM hiện đại mở rộng được context window."""
    B, h, L, d = x.shape
    theta = base ** (-torch.arange(0, d, 2).float() / d)
    pos = torch.arange(L).float()
    freqs = torch.outer(pos, theta)              # (L, d/2)
    cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)

q = torch.randn(1, 8, 16, 64)
print("RoPE output:", rope(q).shape)   # (1, 8, 16, 64)