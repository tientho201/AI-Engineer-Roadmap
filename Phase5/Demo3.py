import timm, torch, torch.nn as nn

# timm = thư viện model ảnh lớn nhất (1000+ kiến trúc pretrained)
print(timm.list_models("convnext*", pretrained=True)[:5])

model = timm.create_model(
    "convnextv2_tiny.fcmae_ft_in22k_in1k",   # mạnh hơn ResNet50 nhiều, cùng tốc độ
    pretrained=True,
    num_classes=10,          # timm tự thay đầu phân loại
    drop_path_rate=0.1,      # stochastic depth — regularization mạnh
)

# Lấy đúng transform mà model được pretrain
cfg = timm.data.resolve_model_data_config(model)
train_tf = timm.data.create_transform(**cfg, is_training=True)
val_tf   = timm.data.create_transform(**cfg, is_training=False)
print(cfg)   # {'input_size': (3,224,224), 'mean': ..., 'std': ...}

# Chiến lược fine-tune 2 giai đoạn
def stage1(model):   # chỉ train đầu phân loại, lr cao
    for p in model.parameters(): p.requires_grad = False
    for p in model.get_classifier().parameters(): p.requires_grad = True
    return torch.optim.AdamW(model.get_classifier().parameters(), lr=1e-3)

def stage2(model):   # mở toàn bộ, lr rất nhỏ, phân tầng
    for p in model.parameters(): p.requires_grad = True
    return torch.optim.AdamW([
        {"params": model.stem.parameters(),    "lr": 1e-5},
        {"params": model.stages.parameters(),  "lr": 5e-5},
        {"params": model.head.parameters(),    "lr": 5e-4},
    ], weight_decay=0.05)