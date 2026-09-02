from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import pandas as pd, numpy as np

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "age":    rng.integers(18, 70, 500).astype(float), # tạo 500 giá trị ngẫu nhiên từ 18 đến 70. Đổi kiểu dữ liệu sang số thực float. Việc này hữu ích nếu sau này cần chứa NaN, vì NaN là giá trị dạng float.
    "income": rng.normal(50_000, 15_000, 500), # Tạo 500 giá trị ngẫu nhiên theo phân phối chuẩn, với trung bình 50_000 và độ lệch chuẩn 15_000. Đây là dữ liệu giả cho income
    "city":   rng.choice(["HCM", "HN", "DN"], 500), # Chọn ngẫu nhiên 500 giá trị thành phố từ danh sách "HCM", "HN", "DN".
})
df.loc[rng.choice(500, 40, replace=False), "income"] = np.nan   # Chọn 500 giá trị ngẫu nhiên từ 500 giá trị và gán giá trị NaN cho 40 giá trị đó.

"""
df.loc: loc chọn dữ liệu theo tên nhãn của dòng và cột. df.loc[row, column]

df.iloc: iloc chọn dữ liệu theo vị trí số nguyên, giống kiểu đếm vị trí trong mảng. df.iloc[row, column]
Ví dụ:
df.iloc[0, 1] Lấy giá trị ở dòng thứ nhất, cột thứ hai.
df.iloc[0:3, 0:2] Lấy giá trị ở dòng thứ 0, 1, 2, cột thứ 0, 1.
"""

y = ((df["income"].fillna(0) > 50_000) & (df["age"] > 30)).astype(int) # Tạo biến y là kết quả của biểu thức logic. (0 và 1)

num_cols, cat_cols = ["age", "income"], ["city"] # Tạo biến num_cols là danh sách các cột số và cat_cols là danh sách các cột categorical.

# Tạo biến preprocess là một ColumnTransformer. ColumnTransformer là một class trong sklearn.compose. ColumnTransformer được sử dụng để áp dụng các transformer cho các cột số và categorical.
preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),("sc",  StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])


model = Pipeline([("prep", preprocess),
                  ("clf",  LogisticRegression(max_iter=1000))])

X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=.2,
                                          stratify=y, random_state=42)
model.fit(X_tr, y_tr)

print(classification_report(y_te, model.predict(X_te)))
print("CV 5-fold:", cross_val_score(model, X_tr, y_tr, cv=5).round(3))