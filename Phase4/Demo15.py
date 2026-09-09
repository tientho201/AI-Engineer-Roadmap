from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI


class Route(BaseModel):
    destination: Literal["ky_thuat", "thanh_toan", "chung"]
    reason: str


class State(TypedDict):
    question: str
    route: str
    answer: str


llm = ChatOpenAI(model="gpt-4o-mini" ,api_key=os.getenv("OPENAI_API_KEY"))


def router(state: State):
    r = llm.with_structured_output(Route).invoke(
        f"Phân loại câu hỏi này về đúng bộ phận: {state['question']}")
    return {"route": r.destination}


def make_expert(name: str, system: str):
    def node(state: State):
        msg = llm.invoke([("system", system), ("user", state["question"])])
        return {"answer": f"[{name}] {msg.content}"}
    return node


g = StateGraph(State)
g.add_node("router", router)
g.add_node("ky_thuat",   make_expert("Kỹ thuật",  "Bạn là kỹ sư hỗ trợ kỹ thuật."))
g.add_node("thanh_toan", make_expert("Thanh toán", "Bạn là chuyên viên thanh toán."))
g.add_node("chung",      make_expert("CSKH",       "Bạn là nhân viên CSKH."))

g.set_entry_point("router")
g.add_conditional_edges("router", lambda s: s["route"],
                        {"ky_thuat": "ky_thuat",
                         "thanh_toan": "thanh_toan",
                         "chung": "chung"})
for n in ["ky_thuat", "thanh_toan", "chung"]:
    g.add_edge(n, END)

app = g.compile()
print(app.invoke({"question": "Tôi bị trừ tiền 2 lần"}))