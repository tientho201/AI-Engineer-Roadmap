import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    """Biến logits thành phân phối xác suất.
    Trừ max để tránh tràn số (numerical stability) — thư viện thật đều làm vậy."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    """CE = -log P(lớp đúng). Chính là -log-likelihood => tối thiểu CE == MLE."""
    n = len(labels)
    return -np.mean(np.log(probs[np.arange(n), labels] + 1e-12))

logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 3.0, 0.2]])
labels = np.array([0, 1])          # nhãn đúng

probs = softmax(logits)
print("Xác suất:\n", probs.round(4))
print("Cross-entropy loss:", round(cross_entropy(probs, labels), 4))
# Mô hình đoán ĐÚNG và TỰ TIN -> loss thấp
# Mô hình đoán SAI và tự tin  -> loss rất cao (phạt nặng)
bad = softmax(np.array([[0.1, 5.0, 0.1]]))
print("Loss khi đoán sai + tự tin:", round(cross_entropy(bad, np.array([0])), 4))