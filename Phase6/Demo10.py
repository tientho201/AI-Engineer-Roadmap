from dataclasses import dataclass, field
from collections import defaultdict

# Giá USD / 1 triệu token (cập nhật lại theo bảng giá hiện hành)
PRICING = {
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00, "cache_read": 0.30},
    "claude-haiku-4-5":  {"in": 0.80, "out": 4.00,  "cache_read": 0.08},
}

@dataclass
class CostTracker:
    by_user: dict = field(default_factory=lambda: defaultdict(float))
    by_feature: dict = field(default_factory=lambda: defaultdict(float))

    def record(self, model, usage, user_id, feature):
        p = PRICING[model]
        cost = (usage.input_tokens        * p["in"]
              + usage.output_tokens       * p["out"]
              + getattr(usage, "cache_read_input_tokens", 0) * p["cache_read"]) / 1e6
        self.by_user[user_id] += cost
        self.by_feature[feature] += cost
        return cost

    def report(self):
        print("Top 5 user tốn kém nhất:")
        for u, c in sorted(self.by_user.items(), key=lambda x: -x[1])[:5]:
            print(f"  {u}: ${c:.4f}")
        print("Chi phí theo tính năng:")
        for f, c in sorted(self.by_feature.items(), key=lambda x: -x[1]):
            print(f"  {f}: ${c:.4f}")

# 3 cách giảm chi phí hiệu quả nhất:
# 1. Prompt caching cho system prompt dài     -> giảm 50-90%
# 2. Model routing: câu dễ -> Haiku, khó -> Sonnet  -> giảm 40-70%
# 3. Semantic cache: câu hỏi lặp -> trả lại kết quả cũ   -> giảm 20-40%