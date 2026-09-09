from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (faithfulness, answer_relevancy,
                          context_precision, context_recall)

data = Dataset.from_dict({
    "question":     ["Tôi được bao nhiêu ngày phép?"],
    "answer":       ["Bạn được 12 ngày phép năm [handbook.pdf:12]."],
    "contexts":     [["Nhân viên chính thức được 12 ngày phép năm."]],
    "ground_truth": ["12 ngày phép năm."],
})

# ragas 0.3.x: mặc định llm_factory -> InstructorLLM bị detect sai là LangChain
# (thiếu run_config) nên gọi agenerate_prompt và crash. Truyền wrapper tường minh.
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

result = evaluate(
    data,
    metrics=[
        faithfulness,        # Câu trả lời có bịa ngoài context không? (chống ảo giác)
        answer_relevancy,    # Có trả đúng trọng tâm câu hỏi không?
        context_precision,   # Chunk lấy về có liên quan không? (đo RANKING)
        context_recall,      # Có lấy đủ chunk cần thiết không? (đo RECALL)
    ],
    llm=llm,
    embeddings=embeddings,
)
print(result)

# CÁCH ĐỌC KẾT QUẢ:
# context_recall thấp    -> lỗi CHUNKING hoặc EMBEDDING (không tìm ra tài liệu)
# context_precision thấp -> lỗi RANKING (tìm ra nhưng xếp sai) -> thêm reranker
# faithfulness thấp      -> lỗi PROMPT (model bịa) -> siết prompt, giảm temperature
# answer_relevancy thấp  -> lỗi sinh câu trả lời -> chỉnh hướng dẫn
