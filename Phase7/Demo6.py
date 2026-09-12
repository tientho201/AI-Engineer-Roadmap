from pydantic import BaseModel, Field
import instructor, subprocess, tempfile, os
from anthropic import Anthropic

client = instructor.from_anthropic(Anthropic())


class CodeSolution(BaseModel):
    reasoning: str
    code: str

class Critique(BaseModel):
    passed: bool
    issues: list[str]
    suggestion: str


def run_tests(code: str, tests: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sol.py")
        open(p, "w").write(code + "\n\n" + tests)
        r = subprocess.run(["python", p], capture_output=True,
                           text=True, timeout=10)
        return r.returncode == 0, (r.stderr or r.stdout)[:2000]


def reflexion_solve(task: str, tests: str, max_iters: int = 4):
    history = []
    for i in range(max_iters):
        prompt = f"Bài toán:\n{task}"
        if history:
            prompt += "\n\nCác lần thử trước đã thất bại:\n" + "\n".join(history)
            prompt += "\n\nPhân tích nguyên nhân rồi viết lại."

        sol = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2048,
            response_model=CodeSolution,
            messages=[{"role": "user", "content": prompt}])

        ok, output = run_tests(sol.code, tests)
        print(f"[lần {i+1}] {'ĐẠT' if ok else 'HỎNG'}")

        if ok:
            return sol.code

        # TỰ PHẢN TỈNH — đây là điểm khác biệt của Reflexion
        crit = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            response_model=Critique,
            messages=[{"role": "user", "content":
                f"Code:\n{sol.code}\n\nLỗi:\n{output}\n\n"
                f"Phân tích nguyên nhân gốc và đề xuất hướng sửa."}])

        history.append(f"Lần {i+1}: {crit.suggestion} | Lỗi: {crit.issues}")

    return None