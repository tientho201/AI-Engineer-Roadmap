import numpy as np

def population_stability_index(expected, actual, bins=10):
    """PSI — metric chuẩn trong ngành tài chính để đo drift."""
    breaks = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breaks[0], breaks[-1] = -np.inf, np.inf

    e = np.histogram(expected, breaks)[0] / len(expected)
    a = np.histogram(actual,   breaks)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


rng = np.random.default_rng(0)
ref = rng.normal(0, 1, 10_000)

for shift, name in [(0, "không drift"), (0.2, "drift nhẹ"), (0.8, "drift nặng")]:
    psi = population_stability_index(ref, rng.normal(shift, 1, 10_000))
    verdict = ("ỔN ĐỊNH" if psi < 0.1 else
               "CẦN THEO DÕI" if psi < 0.25 else "PHẢI TRAIN LẠI")
    print(f"{name:12s} PSI={psi:.4f}  -> {verdict}")

# NGƯỠNG CHUẨN:
#   PSI < 0.10  : ổn định
#   0.10 - 0.25 : thay đổi vừa, theo dõi
#   PSI > 0.25  : thay đổi lớn, cần train lại