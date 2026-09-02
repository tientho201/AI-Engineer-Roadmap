import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data  # (150 , 4)

# Tính trung bình của từng cột
mean_X = X.mean(axis=0) # tính trung bình cột theo hàng dọc mean_col_0 = X[:, 0].sum() / X.shape[0]
print(mean_X)

X_centered = X - mean_X # BẮT BUỘC: PCA yêu cầu dữ liệu đã trừ trung bình

# SVD: X_centered = U @ S @ Vt , phân ra mã trận 
## U.shape   # (150, 4)
## S.shape   # (4,)
## Vt.shape  # (4, 4)
# --- Tự viết bằng SVD ---
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
# print(U.shape, S.shape, Vt.shape)
# print(U)
# print(S)
# print(Vt)
X_pca_manual = X_centered @ Vt[:2].T   # (150, 2)
# X_pca_manual = U[:, :2] * S[:2]
# print(X_pca_manual.shape)
# print(X_pca_manual)

explained = (S**2 / (len(X) - 1)) / np.var(X_centered, axis=0, ddof=1).sum()
# print(explained)
# explained_variance_ratio = S**2 / np.sum(S**2)

# print(explained_variance_ratio)
# print(explained_variance_ratio[:2].sum())
# --- So với sklearn ---
X_pca_sklearn = PCA(n_components=2).fit_transform(X)

print("Tỉ lệ phương sai giải thích:", explained[:2].round(4))  # [0.9246 0.053 ]
print("Khớp với sklearn:", np.allclose(np.abs(X_pca_manual), np.abs(X_pca_sklearn)))  # True