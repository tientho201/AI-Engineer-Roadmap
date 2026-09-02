import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from lightgbm import LGBMClassifier

# ---------- data + model ----------
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
# model = LGBMClassifier(n_estimators=200, verbose=-1).fit(X, y)
print(X.columns)
# # ---------- SHAP Explanation object (API mới từ SHAP ≥ 0.40) ----------
# # TreeExplainer: thuật toán tối ưu cho cây (LightGBM, XGB, sklearn trees)
# # Trả về Explanation object, không phải raw array
# explainer = shap.TreeExplainer(model)
# explanation = explainer(X)   # shape: (n_samples, n_features)

# # explanation.values   : ma trận SHAP values  — (569, 30)
# # explanation.base_values: dự đoán nền (expected_value) — scalar/array
# # explanation.data     : giá trị feature gốc  — (569, 30)

# # ================================================================
# # Biểu đồ 1 — summary_plot (TOÀN CỤC)
# # Trả lời: "feature nào ảnh hưởng mạnh nhất trên toàn bộ dataset?"
# #
# # - Mỗi chấm = 1 sample
# # - Vị trí ngang = SHAP value (+ tăng score, - giảm score)
# # - Màu chấm = giá trị feature đó (đỏ = cao, xanh = thấp)
# # ================================================================
# plt.figure()
# shap.summary_plot(explanation, X, max_display=10, show=False)
# plt.tight_layout()
# plt.savefig("Demo13_summary.png", dpi=150, bbox_inches="tight")
# plt.close()
# print("Saved: Demo13_summary.png")

# # ================================================================
# # Biểu đồ 2 — waterfall_plot (CỤC BỘ — 1 mẫu)
# # Trả lời: "Tại sao model dự đoán sample này ở mức đó?"
# #
# # - Bắt đầu từ base_value (mức nền = trung bình tất cả dự đoán)
# # - Mỗi thanh = đóng góp của 1 feature (đỏ = đẩy lên, xanh = kéo xuống)
# # - Kết thúc ở f(x) = dự đoán thực tế của sample đó
# # ================================================================
# sample_idx = 0   # thay số này để xem mẫu khác
# plt.figure()
# shap.plots.waterfall(explanation[sample_idx], show=False)
# plt.tight_layout()
# plt.savefig("Demo13_waterfall.png", dpi=150, bbox_inches="tight")
# plt.close()
# print(f"Saved: Demo13_waterfall.png  (sample #{sample_idx})")

# # ================================================================
# # In tóm tắt để đọc nhanh
# # ================================================================
# import numpy as np
# mean_abs = np.abs(explanation.values).mean(axis=0)
# top5 = sorted(zip(X.columns, mean_abs), key=lambda x: -x[1])[:5]
# print("\nTop 5 feature theo |SHAP| trung bình:")
# for feat, val in top5:
#     print(f"  {feat:<35} {val:.4f}")
