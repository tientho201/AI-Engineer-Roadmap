"""
TUYỆT ĐỐI KHÔNG dùng exec() hay subprocess trực tiếp cho code do LLM sinh.
Dùng sandbox cô lập: E2B, Modal, Daytona, Cloudflare Workers.
"""
from e2b_code_interpreter import Sandbox

with Sandbox(timeout=60) as sandbox:
    exec_result = sandbox.run_code("""
import pandas as pd
df = pd.read_csv('/data/sales.csv')
print(df.groupby('region')['revenue'].sum())
""")
    print(exec_result.logs.stdout)
    for r in exec_result.results:
        if r.png:                       # biểu đồ matplotlib
            open("chart.png", "wb").write(base64.b64decode(r.png))


# CHECKLIST AN TOÀN cho agent có quyền thực thi:
# [ ] Chạy trong container/VM cô lập, không phải process cha
# [ ] Timeout cứng cho mọi lần thực thi
# [ ] Giới hạn mạng (allowlist domain)
# [ ] Giới hạn RAM/CPU/disk
# [ ] Filesystem chỉ đọc trừ thư mục tạm
# [ ] Không để secret trong biến môi trường của sandbox
# [ ] Ghi log toàn bộ code đã chạy
# [ ] Người duyệt cho thao tác ghi (xóa file, gọi API bên ngoài, chi tiền)