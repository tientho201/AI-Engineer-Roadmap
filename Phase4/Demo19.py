from pydantic import BaseModel, Field
import instructor
from anthropic import Anthropic
import numpy as np

class Judgement(BaseModel):
    reasoning: str = Field(description="Phân tích TRƯỚC khi chấm điểm")
    score: int = Field(ge=1, le=5)
    is_hallucination: bool


JUDGE_PROMPT = """Bạn là giám khảo đánh giá câu trả lời của hệ thống RAG.

THANG ĐIỂM (bám sát, không tự diễn giải):
5 = Hoàn toàn đúng, bám sát context, có trích dẫn
4 = Đúng nhưng thiếu trích dẫn hoặc hơi dài dòng
3 = Đúng một phần, có thiếu sót
2 = Phần lớn sai hoặc lạc đề
1 = Bịa đặt thông tin không có trong context

<context>{context}</context>
<question>{question}</question>
<answer>{answer}</answer>

Phân tích trước, chấm điểm sau."""

client = instructor.from_anthropic(Anthropic())

def judge(context, question, answer, n_votes=3):
    """Lấy mẫu nhiều lần rồi lấy trung vị -> giảm phương sai của judge."""
    scores = []
    for _ in range(n_votes):
        r = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=512, temperature=0.3,
            response_model=Judgement,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                context=context, question=question, answer=answer)}])
        scores.append(r.score)
    return float(np.median(scores)), scores


# HIỆU CHUẨN JUDGE (bước hầu hết team bỏ qua):
# 1. Tự chấm tay 50-100 mẫu -> tạo "golden set"
# 2. Cho judge chấm cùng 50-100 mẫu đó
# 3. Tính Cohen's kappa giữa người và judge
# 4. kappa < 0.6 -> judge KHÔNG ĐÁNG TIN, phải sửa lại prompt chấm điểm
from sklearn.metrics import cohen_kappa_score
# print(cohen_kappa_score(human_scores, judge_scores, weights="quadratic"))