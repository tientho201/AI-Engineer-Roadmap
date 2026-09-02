import torch, torch.nn as nn
from Demo11 import MiniGPT
from torch.utils.data import DataLoader

model = MiniGPT(vocab_size=5000).cuda()

# 1) torch.compile — nhanh hơn 1.3–2x, chỉ cần 1 dòng (PyTorch 2.x)
model = torch.compile(model)

# 2) Gradient accumulation — giả lập batch lớn trên GPU nhỏ
accum_steps = 4       # batch hiệu dụng = batch_size × 4
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step, (xb, yb) in enumerate(DataLoader):     # giả định có dataloader
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(xb.cuda(), yb.cuda())
        loss = loss / accum_steps                # chia để gradient trung bình đúng
    loss.backward()

    if (step + 1) % accum_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

# 3) Gradient checkpointing — đổi tốc độ lấy bộ nhớ (tiết kiệm ~60% VRAM)
from torch.utils.checkpoint import checkpoint
# trong forward: x = checkpoint(block, x, use_reentrant=False)