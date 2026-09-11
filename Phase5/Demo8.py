import torch, torch.nn as nn

def double_conv(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=1, feats=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2)

        c = in_ch
        for f in feats:
            self.downs.append(double_conv(c, f)); c = f

        self.bottleneck = double_conv(feats[-1], feats[-1] * 2)

        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(double_conv(f * 2, f))     # *2 vì nối skip connection

        self.final = nn.Conv2d(feats[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)                          # upsample
            skip = skips[i // 2]
            x = torch.cat([skip, x], dim=1)             # SKIP CONNECTION
            x = self.ups[i + 1](x)                      # <- giữ chi tiết không gian
        return self.final(x)


m = UNet()
print(m(torch.randn(2, 3, 256, 256)).shape)   # (2, 1, 256, 256)


# Loss cho segmentation: kết hợp BCE + Dice là công thức chuẩn
class DiceBCELoss(nn.Module):
    def __init__(self, w_dice=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = w_dice

    def forward(self, logits, target, eps=1e-6):
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(2, 3))
        dice = 1 - (2 * inter + eps) / (p.sum((2,3)) + target.sum((2,3)) + eps)
        return (1 - self.w) * self.bce(logits, target) + self.w * dice.mean()

# Vì sao cần Dice? BCE bị lệch khi vùng cần phân đoạn rất nhỏ so với nền
# (ví dụ: khối u chiếm 1% ảnh). Dice đo trực tiếp độ chồng lấn.