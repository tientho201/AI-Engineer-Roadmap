import torch, torch.nn as nn, torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (B, h, Lq, d)  — "tôi đang tìm gì"
    K: (B, h, Lk, d)  — "tôi chứa thông tin gì"
    V: (B, h, Lk, d)  — "giá trị thực sự được lấy"
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, h, Lq, Lk)
    # Vì sao chia √d_k? Không chia thì khi d_k lớn, scores có phương sai lớn,
    # softmax bị bão hoà -> gradient ≈ 0 -> không học được.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ V, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv  = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        Q, K, V = [t.view(B, L, self.h, self.d_k).transpose(1, 2) for t in qkv]
        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.drop(self.proj(out)), attn


x = torch.randn(2, 10, 512)
mha = MultiHeadAttention()
out, attn = mha(x)
print("output:", out.shape)          # (2, 10, 512)
print("attention map:", attn.shape)  # (2, 8, 10, 10) — mỗi token nhìn vào token nào