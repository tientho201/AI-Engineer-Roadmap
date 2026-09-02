import torch, torch.nn as nn

# CÔNG THỨC VÀNG: out = floor((in + 2*padding - kernel) / stride) + 1
def conv_out(size, kernel, stride=1, padding=0):
    return (size + 2 * padding - kernel) // stride + 1

print(conv_out(224, kernel=7, stride=2, padding=3))   # 112 — giảm nửa
print(conv_out(112, kernel=3, stride=1, padding=1))   # 112 — giữ nguyên ('same')


class ResidualBlock(nn.Module):
    """Khối cốt lõi của ResNet: y = F(x) + x.
    Skip connection cho phép gradient chạy thẳng qua -> train được mạng 100+ tầng."""
    def __init__(self, in_ch, out_ch, stride=1):
        # in_ch: Số channel đầu vào
        # out_ch: Số channel đầu ra
        # stride: Stride conv đầu (1 = giữ size, 2 = giảm một nửa)
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        # Khi đổi shape, nhánh tắt cũng phải đổi theo
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)      # <- CHÍNH LÀ ĐÂY
        return self.relu(out)


x = torch.randn(4, 64, 56, 56)
block = ResidualBlock(64, 128, stride=2)
print(x.shape, "->", block(x).shape)   # torch.Size([4, 128, 28, 28])