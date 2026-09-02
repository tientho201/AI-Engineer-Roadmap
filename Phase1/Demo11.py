import pandas as pd, numpy as np

df = pd.DataFrame({
    "user_id":  [1, 2, 3, 4, 5, 5],
    "age":      [25, np.nan, 35, 120, 28, 28],      # thiếu + ngoại lai
    "city":     ["HCM", "HN", "HCM", "DN", " hcm ", " hcm "],
    "spend":    [100, 250, np.nan, 400, 150, 150],
})

clean = (
    df
    .drop_duplicates(subset="user_id", keep="first")           # trùng lặp
    .assign(city=lambda d: d["city"].str.strip().str.upper())  # chuẩn hoá text
    .query("age.isna() or age < 100", engine="python")         # bỏ ngoại lai
    .fillna({"age": df["age"].median(), "spend": 0})           # điền thiếu
    .astype({"age": "int32"})
)
print(clean)

# Groupby - thao tác dùng nhiều nhất khi phân tích
print(clean.groupby("city").agg(
    n=("user_id", "count"),
    avg_spend=("spend", "mean"),
    max_age=("age", "max"),
))