from dataclasses import dataclass
from typing import Callable

@dataclass
class AgentTrace:
    task: str
    steps: list[dict]         # [{tool, input, output}, ...]
    final_answer: str
    total_tokens: int
    wall_time: float


def evaluate_agent(trace: AgentTrace, expected_tools: set[str],
                   answer_checker: Callable[[str], bool]) -> dict:
    used = {s["tool"] for s in trace.steps}
    return {
        # 1. KẾT QUẢ: có giải đúng không
        "task_success": answer_checker(trace.final_answer),

        # 2. QUỸ ĐẠO: có đi đúng đường không (không chỉ đúng kết quả do may)
        "tool_precision": len(used & expected_tools) / max(len(used), 1),
        "tool_recall":    len(used & expected_tools) / max(len(expected_tools), 1),

        # 3. HIỆU QUẢ: có lãng phí không
        "n_steps": len(trace.steps),
        "tokens_per_task": trace.total_tokens,
        "latency_s": trace.wall_time,

        # 4. ỔN ĐỊNH: có lặp vô ích không
        "repeated_calls": len(trace.steps) - len({
            (s["tool"], str(s["input"])) for s in trace.steps}),
    }


# 4 TẦNG ĐÁNH GIÁ AGENT:
#   Tầng 1 — Unit  : từng tool có chạy đúng không (test thường)
#   Tầng 2 — Step  : mỗi bước có chọn đúng tool + đúng tham số không
#   Tầng 3 — Traj  : toàn bộ quỹ đạo có hợp lý không
#   Tầng 4 — E2E   : kết quả cuối cùng + chi phí + độ trễ
#
# Benchmark tham khảo: tau2-bench, GAIA2, CLEAR
# (ngành đang chuẩn hoá độ tin cậy và độ chính xác chuẩn hoá theo chi phí
#  thành metric hạng nhất, thay vì chỉ nhìn điểm thô)