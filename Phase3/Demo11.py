import torch, torch.nn as nn, torch.nn.functional as F
from Demo8 import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)     # Pre-LN: ổn định hơn Post-LN của paper gốc
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        a, _ = self.attn(self.ln1(x), mask)
        x = x + a                     # residual 1
        x = x + self.ff(self.ln2(x))  # residual 2
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8,
                 n_layers=6, max_len=512, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model) # (vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False) # ( vocab_size , d_model)
        self.head.weight = self.tok_emb.weight     # weight tying -> giảm tham số

    def forward(self, idx, targets=None):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = torch.tril(torch.ones(L, L, device=idx.device)).bool()
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.head(self.ln_f(x)) # (B, L, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.pos_emb.num_embeddings:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


model = MiniGPT(vocab_size=5000)
n_params = sum(p.numel() for p in model.parameters())
print(f"Số tham số: {n_params/1e6:.2f}M")

idx = torch.randint(0, 5000, (2, 20))
logits, loss = model(idx, targets=idx)
print("logits:", logits.shape, "| loss:", round(loss.item(), 4))