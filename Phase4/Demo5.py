import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()   
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1) Khai báo tool — format OpenAI Responses API (khác Anthropic: type + parameters)
tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Lấy thời tiết hiện tại của một thành phố",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Tên thành phố"},
            "unit": {"type": "string", "enum": ["C", "F"], "default": "C"},
        },
        "required": ["city"],
    },
}]

# 2) Hàm thực thi thật
def get_weather(city: str, unit: str = "C") -> dict:
    return {"city": city, "temp": 32, "unit": unit, "condition": "nắng"}

TOOL_MAP = {"get_weather": get_weather}

# 3) VÒNG LẶP AGENT — đây chính là toàn bộ bản chất của một agent
messages = [{"role": "user", "content": "Thời tiết ở Sài Gòn thế nào? Có nên mang ô không?"}]

for turn in range(5):                      # LUÔN giới hạn số vòng, tránh lặp vô hạn
    resp = client.responses.create(
        model="gpt-4o-mini", 
        max_output_tokens=1024,
        tools=tools, 
        input=messages,
    )

    if resp.output[0].type != "function_call":
        print("TRẢ LỜI:", resp.output_text)
        break

    # ghi lại function_call vào lịch sử (đúng format Responses, không phải role/content)
    messages.append({
        "type": "function_call",
        "call_id": resp.output[0].call_id,
        "name": resp.output[0].name,
        "arguments": resp.output[0].arguments,
    })

    # Model muốn gọi tool -> thực thi và trả kết quả về
    results = []
    for block in resp.output:
        if block.type == "function_call":
            print(f"  [gọi tool] {block.name}({block.arguments})")
            try:
                out = TOOL_MAP[block.name](**json.loads(block.arguments))
                results.append({"type": "function_call_output", "call_id": block.call_id,
                                "output": json.dumps(out, ensure_ascii=False)})
            except Exception as e:
                # QUAN TRỌNG: trả lỗi về cho model tự sửa, đừng crash
                results.append({"type": "function_call_output", "call_id": block.call_id,
                                "output": f"Lỗi: {e}"})
    messages.extend(results)
