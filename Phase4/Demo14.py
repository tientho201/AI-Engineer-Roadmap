import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()
@tool
def search_docs(query: str) -> str:
    """Tìm kiếm trong tài liệu nội bộ."""
    return f"Kết quả tìm kiếm cho: {query}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Gửi email. THAO TÁC NGUY HIỂM — cần người duyệt."""
    return f"Đã gửi email tới {to}"


class State(TypedDict):
    messages: Annotated[list, add_messages]


tools = [search_docs, send_email]
llm = ChatOpenAI(model="gpt-4o-mini" ,api_key=os.getenv("OPENAI_API_KEY")).bind_tools(tools)


def agent_node(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

# Checkpointer -> agent nhớ được giữa các lần gọi, khôi phục được khi lỗi
# interrupt_before -> DỪNG để người duyệt trước khi chạy tool nguy hiểm
app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["tools"])

config = {"configurable": {"thread_id":
    "user-123"}}
import json
for event in app.stream({"messages": [("user", "Tìm tài liệu về API rate limit")]},
                        config, stream_mode="values"):
    # stream_mode="values" → mỗi event là dict State, không phải Pydantic model
    print(json.dumps(event, indent=2, ensure_ascii=False, default=str))

# Người duyệt xong thì tiếp tục:
# for event in app.stream(None, config, stream_mode="values"): ...