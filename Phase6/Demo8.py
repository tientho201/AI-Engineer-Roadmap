import pandas as pd, numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

rng = np.random.default_rng(0)

# Dữ liệu tham chiếu (lúc train)
reference = pd.DataFrame({
    "age":    rng.normal(35, 10, 2000),
    "income": rng.normal(50_000, 15_000, 2000),
    "score":  rng.beta(2, 5, 2000),
})

# Dữ liệu production (đã trôi)
current = pd.DataFrame({
    "age":    rng.normal(42, 12, 2000),        # <- tuổi trung bình tăng
    "income": rng.normal(50_000, 15_000, 2000),
    "score":  rng.beta(2, 5, 2000),
})

report = Report([DataDriftPreset(), DataSummaryPreset()])
result = report.run(reference_data=reference, current_data=current)
result.save_html("drift_report.html")

# Trích kết quả để tự động cảnh báo
d = result.dict()
print(d["metrics"][0])