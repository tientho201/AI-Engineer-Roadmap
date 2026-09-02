import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

# Tạo 5,000 dòng dữ liệu, trong đó 97% là giao dịch bình thường (nhãn 0) và chỉ có 3% là gian lận (nhãn 1).
X, y = make_classification(n_samples=5000, weights=[.97, .03], random_state=0)


# Chia dữ liệu thành tập huấn luyện (75%) và tập kiểm tra (25%), đồng thời đảm bảo tỉ lệ mẫu của hai lớp được giữ nguyên trong cả hai tập.
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)

proba = LogisticRegression(max_iter=1000).fit(
    # Huấn luyện mô hình Logistic Regression và dự đoán xác suất thuộc lớp 1 (gian lận).
    Xtr, ytr).predict_proba(Xte)[:, 1]

# Tính toán đường cong Precision-Recall.
prec, rec, thr = precision_recall_curve(yte, proba)

# Tìm ngưỡng đảm bảo bắt được ít nhất 80% ca gian lận
idx = np.where(rec[:-1] >= 0.80)[0][-1]
print(idx)
print(
    f"Ngưỡng = {thr[idx]:.4f} -> Recall={rec[idx]:.2%}, Precision={prec[idx]:.2%}")
# Mặc định 0.5 KHÔNG BAO GIỜ là ngưỡng tối ưu cho dữ liệu mất cân bằng
