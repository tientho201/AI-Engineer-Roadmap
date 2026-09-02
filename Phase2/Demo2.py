import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso

rng = np.random.default_rng(42)
# tạo random phân phối chuẩn N(0, 1) vào matrix 200x5
X = rng.normal(0, 1, (200, 5))
true_w = np.array([3.0, -2.0, 0.0, 0.0, 1.5])   # feature 3,4 là nhiễu
y = X @ true_w + rng.normal(0, 0.5, 200)

# --- Nghiệm đóng: w = (XᵀX)⁻¹Xᵀy ---
# Nếu gộp cột toàn số 1 vào (X), nghiệm normal equation là w[1:] vì w[0] là hệ số của cột 1
Xb = np.c_[np.ones(len(X)), X]
w_closed = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
print("Nghiệm đóng :", w_closed[1:].round(3))

# --- So sánh regularization ---
for name, model in [("OLS  ", LinearRegression()),
                    ("Ridge", Ridge(alpha=10)),
                    ("Lasso", Lasso(alpha=0.3))]:
    model.fit(X, y)
    print(f"{name}: {model.coef_.round(3)}")
# Lasso đưa hệ số của feature nhiễu về ĐÚNG 0 -> tự động chọn feature
# Ridge chỉ co nhỏ lại, không bằng 0
