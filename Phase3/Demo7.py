import torch, torch.nn as nn

seq_len, hidden = 100, 32
rnn  = nn.RNN(hidden, hidden, batch_first=True)
lstm = nn.LSTM(hidden, hidden, batch_first=True)

for name, model in [("RNN ", rnn), ("LSTM", lstm)]:
    x = torch.randn(1, seq_len, hidden, requires_grad=True)
    out, _ = model(x)
    out[:, -1].sum().backward()
    # Gradient tại bước đầu tiên so với bước cuối
    g_first = x.grad[0, 0].norm().item()
    g_last  = x.grad[0, -1].norm().item()
    print(f"{name}: grad đầu chuỗi={g_first:.2e}  cuối chuỗi={g_last:.2e}  "
          f"tỉ lệ={g_first/g_last:.2e}")

# RNN: gradient ở đầu chuỗi gần như triệt tiêu -> không học được phụ thuộc xa
# LSTM: cổng quên giữ gradient sống lâu hơn nhiều