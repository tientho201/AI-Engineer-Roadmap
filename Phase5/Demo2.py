import albumentations as A
import cv2

img_bgr = cv2.imread("photo.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

train_tf = A.Compose([
    A.LongestMaxSize(max_size=640),
    A.PadIfNeeded(640, 640, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),          # quan trọng cho camera giám sát
    A.ISONoise(p=0.2),                          # giả lập nhiễu camera đêm
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                            min_visibility=0.3))
# <- bbox_params giúp bounding box TỰ ĐỘNG biến đổi theo ảnh.
#    Tự viết tay phần này rất dễ sai.

aug = train_tf(image=img_rgb,
               bboxes=[[0.5, 0.5, 0.2, 0.3]],
               class_labels=[0])
print(aug["image"].shape, aug["bboxes"])