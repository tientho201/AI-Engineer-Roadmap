import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np


def sigmoid(z):
    # clip chống tràn số (-500 < z < 500)
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class MyLogisticRegression:
    def __init__(self, lr=0.1, epochs=1000, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)  # (n,d) x (d,) = (n,)
            # Gradient của log-loss đơn giản đẹp bất ngờ: (p - y)
            # (d,n) x (n,) + (d,) = (d,)
            dw = X.T @ (p - y) / n + self.l2 * self.w
            db = np.mean(p - y)  # scalar
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


X, y = make_classification(n_samples=1000, n_features=10, random_state=0)
# print(X.shape, y.shape)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=0)

m = MyLogisticRegression(lr=0.5, epochs=2000).fit(Xtr, ytr)
print("Accuracy:", accuracy_score(yte, m.predict(Xte)))
print("ROC-AUC :", roc_auc_score(yte, m.predict_proba(Xte)).round(4))


fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(yte, m.predict_proba(Xte), alpha=0.1)
ax.set_xlabel('True label')
ax.set_ylabel('Predicted probability')
ax.set_title('ROC curve')
plt.tight_layout()
plt.show()
plt.savefig("roc_curve.png")
