from pydantic import BaseModel, Field
from typing import List, Optional
from app.schemas.mongo_chat import TokenUsage

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message author: 'user' or 'assistant'")
    content: str = Field(..., description="The message string content")

class ChatRequest(BaseModel):
    message: str = Field(..., description="Current message query from the user")
    history: List[ChatMessage] = Field(default_factory=list, description="Lịch sử hội thoại trước đó (client-side state)")
    provider: Optional[str] = Field(default=None, description="Ghi đè nhà cung cấp LLM. Hỗ trợ: 'openai', 'gemini', 'ollama'. Mặc định sử dụng LLM_PROVIDER từ .env.")
    model: Optional[str] = Field(
        default=None, 
        description=(
            "Ghi đè tên model tương ứng. Hỗ trợ: "
            "- 'openai': 'gpt-4o-mini' (mặc định), 'gpt-4o', 'gpt-4', v.v. "
            "- 'gemini': 'gemini-2.0-flash' (mặc định), 'gemini-2.5-flash', 'gemini-3.5-flash', 'gemini-flash-latest', v.v. "
            "- 'ollama': 'qwen2.5:7b' (mặc định), 'llama3', 'qwen2.5:14b', v.v. (tùy thuộc vào các model đã pull cục bộ)."
        )
    )
    temperature: Optional[float] = Field(default=None, description="Ghi đè nhiệt độ sáng tạo (0.0 đến 1.0). Mặc định lấy từ cấu hình LLM_TEMPERATURE từ .env.")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Phản hồi từ AI Agent")
    status: str = Field(default="success", description="Trạng thái phản hồi (success, error)")
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Báo cáo số token tiêu thụ")
