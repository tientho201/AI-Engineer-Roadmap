import torch 

# requires_grad=True -> PyTorch ghi lại mọi phép toán vào đồ thị tính toán
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([-3.0], requires_grad=True)
b = torch.tensor([6.88], requires_grad=True)

y = w * x + b
print("y =", y)

o = torch.tanh(y)
o.backward()

print(f"o = {o.item():.4f}")
print(f"dL/dw = {w.grad.item():.4f}")   # khớp với Demo 5 của Phase 1
print(f"dL/dx = {x.grad.item():.4f}")

# 3 điều BẮT BUỘC nhớ:
# 1) grad TÍCH LUỸ, không ghi đè -> phải optimizer.zero_grad() mỗi bước
w.grad.zero_()
# 2) Tắt autograd khi suy luận -> tiết kiệm ~50% bộ nhớ
with torch.no_grad():
    _ = torch.tanh(x * w + b)
# 3) .detach() cắt tensor khỏi đồ thị (dùng khi log giá trị)
loss_value = o.detach().cpu().item()