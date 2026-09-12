import json
from fastapi.responses import StreamingResponse
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

@app.post("/chat/stream")
async def chat_stream(body: dict):
    async def event_generator():
        try:
            async with client.messages.stream(
                model="claude-sonnet-4-6", max_tokens=2048,
                messages=body["messages"],
            ) as stream:
                async for text in stream.text_stream:
                    yield f"data: {json.dumps({'delta': text})}\n\n"
                final = await stream.get_final_message()
                yield f"data: {json.dumps({'done': True, 'usage': {'in': final.usage.input_tokens, 'out': final.usage.output_tokens}})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})   # tắt buffer của nginx

# TTFT (Time To First Token) là metric UX quan trọng nhất.
# Người dùng chấp nhận 20s tổng thời gian nếu chữ xuất hiện sau 0.5s,
# nhưng KHÔNG chấp nhận chờ 5s màn hình trắng rồi nhận hết một lúc.