"""MiniGPT v2 — RoPE attention, không dùng positional embedding học được."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope(x, base=10000):
    B, h, L, d = x.shape
    theta = base ** (-torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    pos = torch.arange(L, device=x.device, dtype=x.dtype)
    freqs = torch.outer(pos, theta)
    cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ V, attn


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        Q, K, V = [t.view(B, L, self.h, self.d_k).transpose(1, 2) for t in qkv]
        Q, K = rope(Q), rope(K)
        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.drop(self.proj(out)), attn


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionRoPE(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        a, attn = self.attn(self.ln1(x), mask)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, attn


class MiniGPTRoPE(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8,
                 n_layers=6, max_len=512, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None, return_attn=False):
        B, L = idx.shape
        x = self.drop(self.tok_emb(idx))
        mask = torch.tril(torch.ones(L, L, device=idx.device)).bool()
        attn_maps = []
        for blk in self.blocks:
            x, attn = blk(x, mask)
            if return_attn:
                attn_maps.append(attn)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_attn:
            return logits, loss, attn_maps
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
