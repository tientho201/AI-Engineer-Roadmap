import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers=3, dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(d, hidden),
                nn.LayerNorm(hidden),       # ổn định hơn BatchNorm cho batch nhỏ
                nn.GELU(),                  # mượt hơn ReLU, chuẩn của Transformer
                nn.Dropout(dropout),
            ]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = MLP(784, 256, 10)
from torchinfo import summary
summary(model, input_size=(32, 784))