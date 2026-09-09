"""weather_mcp.py — MCP server tối giản. Chạy: uv run weather_mcp.py"""
from mcp.server.mcpserver import MCPServer
import httpx

mcp = MCPServer("weather")


@mcp.tool()
async def get_forecast(city: str, days: int = 3) -> dict:
    """Lấy dự báo thời tiết cho một thành phố.

    Args:
        city: Tên thành phố, ví dụ 'Ho Chi Minh City'
        days: Số ngày dự báo (1-7)
    """
    # Docstring NÀY chính là thứ LLM đọc để quyết định gọi tool.
    # Viết docstring tệ = agent gọi tool sai.
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.example.com/forecast",
                             params={"city": city, "days": days})
        return r.json()


@mcp.resource("config://units")
def get_units() -> str:
    """Đơn vị đo đang dùng."""
    return "celsius"


if __name__ == "__main__":
    mcp.run(transport="stdio")