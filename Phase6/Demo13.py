"""locustfile.py — chạy: locust -f locustfile.py --host http://localhost:8000"""
from locust import HttpUser, task, between
import random

class APIUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(10)
    def predict(self):
        payload = {"features": [[random.random() for _ in range(30)]]}
        with self.client.post("/predict", json=payload, catch_response=True) as r:
            if r.elapsed.total_seconds() > 1.0:
                r.failure("Quá chậm (>1s)")

    @task(1)
    def health(self):
        self.client.get("/health")

# Cần biết trước khi lên production:
# - QPS tối đa trước khi p95 vượt SLA
# - Bao nhiêu replica cho traffic dự kiến
# - Hệ thống hỏng như thế nào khi quá tải (giảm dần hay sụp hoàn toàn)