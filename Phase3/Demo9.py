import torch

L = 6
# Ma trận tam giác dưới: token thứ i chỉ được nhìn token 0..i
causal = torch.tril(torch.ones(L, L))
print(causal.int())
# [[1,0,0,0,0,0],
#  [1,1,0,0,0,0],
#  [1,1,1,0,0,0],
#  ...]
# Đây chính là khác biệt duy nhất về kiến trúc giữa
# BERT (nhìn 2 chiều) và GPT (chỉ nhìn quá khứ).

# Padding mask: bỏ qua token đệm
tokens = torch.tensor([[5, 8, 2, 0, 0, 0]])   # 0 = <pad> # (1, 6)
pad_mask = (tokens != 0).unsqueeze(1).unsqueeze(2) # (1, 1, 1, 6)
combined = causal.bool() & pad_mask # (1, 1, 6, 6)
print(combined.shape)   # (1, 1, 6, 6)