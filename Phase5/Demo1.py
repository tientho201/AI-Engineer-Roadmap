import cv2
import numpy as np
from PIL import Image
import torch

img_bgr = cv2.imread("photo.jpg")

# CẠM BẪY 1: OpenCV đọc ảnh ở BGR, mọi model deep learning dùng RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# Quên dòng này -> model vẫn chạy, không báo lỗi, nhưng độ chính xác tụt 5-15%.

# CẠM BẪY 2: thứ tự chiều
print(img_bgr.shape)              # (H, W, C) — OpenCV / NumPy
# PyTorch cần     (C, H, W)
# Batch PyTorch   (B, C, H, W)
tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

# CẠM BẪY 3: resize làm méo tỉ lệ
h, w = img_rgb.shape[:2]
bad  = cv2.resize(img_rgb, (640, 640)) # méo hình

def letterbox(img, size=640, color=(114, 114, 114)):
    """Giữ nguyên tỉ lệ, đệm xám — chuẩn của YOLO."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = cv2.copyMakeBorder(resized, top, size - nh - top,
                             left, size - nw - left,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)     # trả r và offset để map box về ảnh gốc

good, ratio, pad = letterbox(img_rgb)