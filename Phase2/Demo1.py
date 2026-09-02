# Demo 1 — Nhìn thấy bias–variance bằng mắt
import numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 1, 30))[:, None] # [:, None] -> thêm chiều để phù hợp với sklearn
y = np.sin(2 * np.pi * X).ravel() + rng.normal(0, 0.2, 30)
X_test = np.linspace(0, 1, 200)[:, None] # [:, None] -> thêm chiều để phù hợp với sklearn. Tạo ra 200 điểm cách đều nhau trên đoạn [0, 1].

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, deg in zip(axes, [1, 4, 20]):
    m = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(X, y) # Tạo pipeline gồm PolynomialFeatures và LinearRegression. PolynomialFeatures(deg) -> tạo ra đa thức bậc deg. LinearRegression() -> tạo ra mô hình tuyến tính.
    ax.scatter(X, y, c="k", s=20) # Vẽ điểm dữ liệu.
    ax.plot(X_test, m.predict(X_test), c="r") # Vẽ đường cong mô hình.
    ax.plot(X_test, np.sin(2*np.pi*X_test), c="g", ls="--", label="hàm thật") # Vẽ đường cong thật.
    ax.set_ylim(-2, 2) # Đặt khoảng giá trị của trục y.
    ax.set_title(f"bậc {deg} | train MSE={mean_squared_error(y, m.predict(X)):.4f}") # Đặt tiêu đề cho đồ thị.
    ax.legend() # Hiển thị legend.
    
# bậc 1  -> underfit (bias cao)
# bậc 4  -> vừa đẹp
# bậc 20 -> overfit (variance cao): train MSE gần 0 nhưng đường cong điên loạn
plt.savefig("Demo1.png")
plt.tight_layout(); plt.show()
plt.close()