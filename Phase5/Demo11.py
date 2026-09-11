import base64
from anthropic import Anthropic

client = Anthropic()

with open("invoice.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": img_b64}},
            {"type": "text", "text":
                "Trích xuất thông tin hóa đơn thành JSON: "
                "{số_hóa_đơn, ngày, nhà_cung_cấp, tổng_tiền, danh_sách_mặt_hàng[]}. "
                "Chỉ trả JSON, không giải thích."},
        ],
    }],
)
print(resp.content[0].text)

# BÀI HỌC 2026: nhiều bài toán CV trước đây cần
# OCR + layout detection + rule engine (vài tuần code)
# giờ VLM giải được trong 1 prompt.
#
# NHƯÕNG khi nào vẫn phải dùng model chuyên dụng:
# - Cần thời gian thực (>15 FPS)      -> YOLO
# - Chạy offline / trên thiết bị biên -> model nhỏ
# - Khối lượng lớn, chi phí nhạy cảm  -> model tự host
# - Cần độ chính xác pixel (mask)      -> segmentation model