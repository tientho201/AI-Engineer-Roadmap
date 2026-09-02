import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms.v2 as T

weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)

# GIAI ĐOẠN 1: đóng băng backbone, chỉ train đầu phân loại
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)   # layer mới mặc định requires_grad=True

# GIAI ĐOẠN 2: mở băng dần từ tầng cuối, lr rất nhỏ
def unfreeze_last(model, n_blocks=1):
    for name, p in model.named_parameters():
        if any(f"layer{4 - i}" in name for i in range(n_blocks)):
            p.requires_grad = True

# Discriminative learning rate: tầng sâu học chậm, tầng nông học nhanh
optimizer = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(),     "lr": 1e-3},
])

# Augmentation — nguồn cải thiện lớn nhất khi ít dữ liệu
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.TrivialAugmentWide(),                  # augmentation tự động, rất mạnh
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.25),
])