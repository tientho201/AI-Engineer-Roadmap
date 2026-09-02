import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

# Bài toán gian lận thẻ: 99% giao dịch bình thường, 1% gian lận
y_true = np.array([0]*990 + [1]*10)


# model lười: đoán tất cả "không gian lận"
y_lazy = np.zeros(1000)
y_good = np.array([0]*985 + [1]*5 + [1]*8 + [0]*2)  # model thật

for name, pred in [("Model lười", y_lazy), ("Model thật ", y_good)]:
    print(f"{name} | Acc={accuracy_score(y_true, pred):.3f} "
          f"Precision={precision_score(y_true, pred, zero_division=0):.3f} "
          f"Recall={recall_score(y_true, pred):.3f} "
          f"F1={f1_score(y_true, pred):.3f}")

# Model lười đạt Accuracy 99% nhưng Recall = 0 -> VÔ DỤNG HOÀN TOÀN
