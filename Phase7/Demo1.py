import torch
import numpy as np

def quantize_int8_symmetric(w: torch.Tensor):
    """Quantization đối xứng: fp32 -> int8. Đây là nền tảng của mọi kỹ thuật khác."""
    scale = w.abs().max() / 127.0
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale

def dequantize(q: torch.Tensor, scale: float):
    return q.float() * scale


w = torch.randn(1024, 1024)
q, s = quantize_int8_symmetric(w)
w_hat = dequantize(q, s)

print(f"Bộ nhớ fp32: {w.numel() * 4 / 1e6:.2f} MB")
print(f"Bộ nhớ int8: {q.numel() * 1 / 1e6:.2f} MB  (giảm 4x)")
print(f"Sai số trung bình: {(w - w_hat).abs().mean():.6f}")
print(f"Sai số tối đa    : {(w - w_hat).abs().max():.6f}")


# VẤN ĐỀ THỰC TẾ: OUTLIER
# Một vài trọng số cực lớn sẽ "kéo" scale lên, làm toàn bộ trọng số
# còn lại bị dồn vào vài mức lượng tử -> mất thông tin nghiêm trọng.
w_outlier = w.clone()
w_outlier[0, 0] = 100.0
q2, s2 = quantize_int8_symmetric(w_outlier)
print(f"\nScale bình thường: {s:.6f}")
print(f"Scale có outlier : {s2:.6f}  (lớn hơn {s2/s:.0f}x!)")
print(f"Sai số tăng      : {(w_outlier - dequantize(q2,s2)).abs().mean():.6f}")

# ĐÂY CHÍNH LÀ VẤN ĐỀ MÀ CÁC KỸ THUẬT SAU GIẢI QUYẾT:
# - AWQ: giữ nguyên độ chính xác cho ~1% kênh quan trọng (dựa trên activation)
# - GPTQ: quantize từng cột, bù sai số sang cột còn lại (dùng Hessian)
# - SmoothQuant: chuyển "độ khó" từ activation sang weight
# - NF4 (QLoRA): dùng phân phối chuẩn thay vì chia đều