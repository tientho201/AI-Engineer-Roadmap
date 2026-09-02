import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# X: mảng 2D chứa dữ liệu đầu vào (n_samples, n_features)
# k: số cụm muốn tìm
# iters: số vòng lặp tối đa
# seed: seed để đảm bảo kết quả lặp lại


def my_kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        # B1: gán mỗi điểm về tâm gần nhất
        d = np.linalg.norm(X[:, None] - centers[None], axis=2)   # (n, k)
        labels = d.argmin(axis=1)
        # B2: cập nhật tâm = trung bình các điểm trong cụm
        new = np.array([X[labels == i].mean(axis=0) if (labels == i).any()
                        else centers[i] for i in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    inertia = ((X - centers[labels]) ** 2).sum()
    return labels, centers, inertia


X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

print(" k | inertia  | silhouette")
for k in range(2, 8):
    labels, _, inertia = my_kmeans(X, k)
    print(f"{k:2d} | {inertia:8.1f} | {silhouette_score(X, labels):.4f}")
# Silhouette cao nhất tại k=4 -> đúng số cụm thật
