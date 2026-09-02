import numpy as np

# Bài toán: hồi quy tuyến tính y = 3x + 5, tìm lại w=3, b=5 từ dữ liệu
np.random.seed(42)
X = np.random.rand(200, 1) * 10  # Tạo ma trân 200x1 sau đó nhân từng phần tử với 10
y = 3 * X + 5 + np.random.randn(200, 1) * 0.5
# noise = np.random.randn(200, 1) * 0.5: tạo ra ma trận 200x1 với các phần tử là các số ngẫu nhiên từ phân phối chuẩn với độ lệch chuẩn 0.5 sẽ có tác dụng làm cho dữ liệu có noise, tức là dữ liệu sẽ không phải là một đường thẳng hoàn hảo mà sẽ có một số điểm dữ liệu bị lệch khỏi đường thẳng.

w, b = 0.0, 0.0
lr, epochs, n = 0.01, 1000, len(X)

for ep in range(epochs):
    y_pred = w * X + b
    loss = np.mean((y_pred - y) ** 2)          # MSE

    # Đạo hàm MSE:  dL/dw = 2/n * Σ (ŷ-y)·x  ,  dL/db = 2/n * Σ (ŷ-y)
    dw = (2 / n) * np.sum((y_pred - y) * X)
    db = (2 / n) * np.sum(y_pred - y)

    w -= lr * dw    # bước đi ngược hướng gradient
    b -= lr * db

    if ep % 200 == 0:
        print(f"epoch {ep:4d} | loss={loss:7.4f} | w={w:.3f} b={b:.3f}")

print(f"\nKết quả: w={w:.3f} (thật: 3.0), b={b:.3f} (thật: 5.0)")