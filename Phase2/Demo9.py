import pandas as pd
import numpy as np

df = pd.DataFrame({
    "ngay_mua":  pd.to_datetime(["2026-01-15", "2026-03-22", "2026-07-04", "2026-12-25"]),
    "gia":       [100_000, 2_500_000, 50_000, 890_000],
    "danh_muc":  ["điện tử", "điện tử", "thực phẩm", "thời trang"],
    "so_luong":  [1, 2, 10, 1],
})

# 1) Tách đặc trưng thời gian
df["thang"] = df["ngay_mua"].dt.month
df["thu_trong_tuan"] = df["ngay_mua"].dt.dayofweek
df["la_cuoi_tuan"] = (df["thu_trong_tuan"] >= 5).astype(int)
# Mã hoá chu kỳ: tháng 12 và tháng 1 phải "gần nhau"
df["thang_sin"] = np.sin(2 * np.pi * df["thang"] / 12)
df["thang_cos"] = np.cos(2 * np.pi * df["thang"] / 12)

# 2) Biến đổi log cho phân phối lệch
df["log_gia"] = np.log1p(df["gia"])  # log(1+x)  để tránh log(0)

# 3) Feature tương tác
df["tong_tien"] = df["gia"] * df["so_luong"]
df["gia_tren_dv"] = df["tong_tien"] / df["so_luong"]

# 4) Biến một giá trị số liên tục thành biến phân loại theo từng khoảng
df["phan_khuc"] = pd.cut(df["gia"], bins=[0, 200_000, 1_000_000, np.inf],
                         labels=["rẻ", "trung bình", "cao cấp"])

print(df)
