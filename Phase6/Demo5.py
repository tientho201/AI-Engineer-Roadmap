from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B-Instruct",
    messages=[{"role": "user", "content": "Giải thích PagedAttention"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)