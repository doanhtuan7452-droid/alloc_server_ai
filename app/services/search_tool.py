from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from app.services.search_strategies import SearchStrategyFactory
from app.core.config import settings

class SearchInput(BaseModel):
    query: str = Field(..., description="Từ khóa hoặc câu hỏi cần tìm kiếm trên internet")

async def internet_search_func(query: str) -> str:
    """Công cụ tìm kiếm thông tin trên internet."""
    strategy = SearchStrategyFactory.get_strategy()
    max_results = settings.SEARCH_MAX_RESULTS
    results = await strategy.search(query, max_results=max_results)
    
    if not results:
        return f"Không tìm thấy kết quả nào trên internet cho câu truy vấn: '{query}'"
    
    formatted_results = []
    for idx, item in enumerate(results, 1):
        formatted_results.append(
            f"{idx}. Tiêu đề: {item['title']}\n"
            f"   Liên kết: {item['link']}\n"
            f"   Nội dung: {item['snippet']}\n"
        )
    
    return "\n".join(formatted_results)

internet_search_tool = StructuredTool.from_function(
    coroutine=internet_search_func,
    name="internet_search",
    description=(
        "Sử dụng công cụ này khi người dùng yêu cầu tìm kiếm thông tin mới nhất trên internet "
        "hoặc các thông tin ngoài hệ thống, kiến thức chung mà AI không tự có sẵn."
    ),
    args_schema=SearchInput
)
