from transformers import SamModel, SamProcessor
import torch
from PIL import Image

processor = SamProcessor.from_pretrained("facebook/sam2-hiera-large")
model = SamModel.from_pretrained("facebook/sam2-hiera-large").cuda()

image = Image.open("scene.jpg")
input_points = [[[450, 600]]]        # chỉ cần click 1 điểm vào vật thể

inputs = processor(image, input_points=input_points, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs, multimask_output=True)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu())
print(masks[0].shape)     # (1, 3, H, W) — 3 giảng thuyết mặt nạ
print("điểm tin cậy:", outputs.iou_scores)

# Ứng dụng thực tế: dùng SAM để TỰ ĐỘNG GÁN NHÃN dataset
# -> YOLO detect ra box -> SAM biến box thành mask chính xác
# -> tiết kiệm hàng trăm giờ gán nhãn thủ công.