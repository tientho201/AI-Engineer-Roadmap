import numpy as np, time

N = 1_000_000
a, b = np.random.rand(N), np.random.rand(N)


"""
Muốn tạo số trong khoảng khác
Vì np.random.rand chỉ sinh số trong [0, 1), ta có thể biến đổi nó.
Tạo số trong khoảng [0, 10):
    x = np.random.rand() * 10
Tạo mảng số trong khoảng [5, 15):
    x = 5 + np.random.rand(3, 4) * 10
Công thức tổng quát:
    a + np.random.rand(...) * (b - a)
sẽ cho số trong khoảng [a, b).

"""
# Cách 1: vòng lặp Python
t0 = time.perf_counter()
result_loop = [a[i] * b[i] for i in range(N)]
t_loop = time.perf_counter() - t0

# Cách 2: vectorized NumPy
t0 = time.perf_counter()
result_vec = a * b
t_vec = time.perf_counter() - t0

print(f"Vòng lặp : {t_loop*1000:8.2f} ms")
print(f"NumPy    : {t_vec*1000:8.2f} ms")
print(f"Nhanh hơn: {t_loop/t_vec:.0f}x")     # thường 100-300