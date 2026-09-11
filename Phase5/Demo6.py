from ultralytics import YOLO

# Cấu trúc dataset bắt buộc:
# dataset/
#   images/train/*.jpg   labels/train/*.txt
#   images/val/*.jpg     labels/val/*.txt
#   data.yaml
#
# Mỗi dòng trong file .txt:  class_id  x_center  y_center  width  height
# TẤT CẢ đều chuẩn hoá về [0, 1] theo kích thước ảnh.

model = YOLO("yolo11s.pt")     # n/s/m/l/x — bắt đầu bằng 's'

results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,             # early stopping
    optimizer="AdamW",
    lr0=1e-3,
    cos_lr=True,
    # Augmentation — chỉnh theo bài toán thực tế
    mosaic=1.0,              # ghép 4 ảnh — cực mạnh cho object nhỏ
    close_mosaic=10,         # tắt mosaic 10 epoch cuối để model ổn định
    mixup=0.1,
    degrees=10, translate=0.1, scale=0.5, fliplr=0.5,
    amp=True,
    project="runs", name="exp1",
)

# Đánh giá
metrics = model.val()
print(f"mAP50    = {metrics.box.map50:.4f}")
print(f"mAP50-95 = {metrics.box.map:.4f}")

# Xuất để triển khai
model.export(format="onnx", dynamic=True, simplify=True)
model.export(format="engine", half=True)     # TensorRT — nhanh nhất trên NVIDIA