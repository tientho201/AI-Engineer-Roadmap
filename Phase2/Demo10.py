from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold,
                                     TimeSeriesSplit)
import numpy as np

X = np.arange(20).reshape(-1, 1)
y = np.array([0]*15 + [1]*5)
groups = np.repeat(np.arange(5), 4)     # 5 bệnh nhân, mỗi người 4 mẫu

print("StratifiedKFold — giữ tỉ lệ lớp (mặc định cho phân loại):")
for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
    print("  test:", te, "| tỉ lệ lớp 1:", y[te].mean().round(2))

print("\nGroupKFold — không cho cùng 1 bệnh nhân xuất hiện ở cả train và test:")
for tr, te in GroupKFold(3).split(X, y, groups):
    print("  test groups:", np.unique(groups[te]))

print("\nTimeSeriesSplit — KHÔNG BAO GIỜ train trên tương lai:")
for tr, te in TimeSeriesSplit(3).split(X):
    print(f"  train={tr[0]}..{tr[-1]}  test={te[0]}..{te[-1]}")
