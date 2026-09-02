from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=500, noise=0.06, random_state=0)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(*X.T, c=KMeans(2, n_init=10,
              random_state=0).fit_predict(X), cmap="coolwarm", s=12)
ax[0].set_title("K-Means — SAI (giả định cụm hình cầu)")
ax[1].scatter(*X.T, c=DBSCAN(eps=0.2, min_samples=5).fit_predict(X),
              cmap="coolwarm", s=12)
ax[1].set_title("DBSCAN — ĐÚNG (dựa trên mật độ)")
plt.tight_layout()
plt.savefig("demo8.png")
plt.show()
