"""
RAG thường thất bại với câu hỏi kiểu:
  "Những người nào từng làm việc ở cả công ty A và công ty B?"
  "Chủ đề xuyên suốt 200 báo cáo này là gì?"
Vì câu trả lời không nằm trong BẤT KỲ chunk đơn lẻ nào.
"""
from pydantic import BaseModel
import instructor, networkx as nx
from anthropic import Anthropic

class Entity(BaseModel):
    name: str
    type: str        # PERSON / ORG / PRODUCT / CONCEPT

class Relation(BaseModel):
    source: str
    target: str
    relation: str
    evidence: str

class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


client = instructor.from_anthropic(Anthropic())

def build_graph(chunks: list[str]) -> nx.DiGraph:
    G = nx.DiGraph()
    for chunk in chunks:
        ex = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2048,
            response_model=Extraction,
            messages=[{"role": "user", "content":
                f"Trích xuất thực thể và quan hệ từ đoạn sau:\n\n{chunk}"}])
        for e in ex.entities:
            G.add_node(e.name, type=e.type)
        for r in ex.relations:
            G.add_edge(r.source, r.target,
                       relation=r.relation, evidence=r.evidence)
    return G


def graph_query(G: nx.DiGraph, a: str, b: str):
    """Truy vấn multi-hop mà vector search không làm được."""
    try:
        path = nx.shortest_path(G.to_undirected(), a, b)
        return [(u, G.get_edge_data(u, v) or G.get_edge_data(v, u), v)
                for u, v in zip(path, path[1:])]
    except nx.NetworkXNoPath:
        return []


# Phát hiện cộng đồng -> tóm tắt từng cụm -> trả lời câu hỏi tổng quan
def community_summaries(G, client):
    import networkx.algorithms.community as nx_comm
    communities = nx_comm.louvain_communities(G.to_undirected())
    summaries = []
    for com in communities:
        sub = G.subgraph(com)
        facts = [f"{u} --{d['relation']}--> {v}" for u, v, d in sub.edges(data=True)]
        summaries.append({"members": list(com), "facts": facts})
    return summaries


# KHI NÀO DÙNG GRAPHRAG:
#   ✓ Câu hỏi multi-hop ("ai liên quan đến ai qua cái gì")
#   ✓ Câu hỏi tổng quan toàn corpus ("chủ đề chính là gì")
#   ✗ Câu hỏi tra cứu đơn giản -> hybrid search rẻ hơn nhiều
# Chi phí xây graph rất cao (1 lần gọi LLM / chunk). Cân nhắc kỹ.