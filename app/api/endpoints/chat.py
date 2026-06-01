from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent import agent_service

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post(
    "/agent/query", 
    response_model=ChatResponse,
    summary="Truy vấn AI Agent (hỗ trợ chuyển đổi linh hoạt Model/Provider)",
    description=(
        "Gửi câu hỏi tới AI Agent để trò chuyện hoặc yêu cầu phân bổ nhân viên, "
        "dự báo rủi ro dự án. Cổng API hỗ trợ các nhà cung cấp (providers) và mô hình (models) sau:\n\n"
        "1. **OpenAI (`provider: \"openai\"`)**:\n"
        "   - Các model hỗ trợ: `gpt-4o-mini` (mặc định), `gpt-4o`, `gpt-4`, v.v.\n"
        "   - Yêu cầu cấu hình `OPENAI_API_KEY` trong file `.env`.\n\n"
        "2. **Google Gemini (`provider: \"gemini\"`)**:\n"
        "   - Các model hỗ trợ: `gemini-2.0-flash` (mặc định), `gemini-2.5-flash`, `gemini-3.5-flash`, `gemini-flash-latest`, v.v. (Lưu ý: model cũ `gemini-1.5-flash` và `gemini-pro` có thể đã bị ngừng hỗ trợ bởi Google AI Studio).\n"
        "   - Yêu cầu cấu hình `GEMINI_API_KEY` hoặc `GOOGLE_API_KEY` trong file `.env`.\n\n"
        "3. **Ollama (`provider: \"ollama\"`)**:\n"
        "   - Các model hỗ trợ: `qwen2.5:7b` (mặc định), `llama3`, `qwen2.5:14b`, v.v. (tùy thuộc vào danh sách mô hình đã được tải về máy chạy Ollama).\n"
        "   - Địa chỉ kết nối mặc định: `http://localhost:11434` (tùy biến qua `OLLAMA_BASE_URL` trong `.env`)."
    )
)
async def query_agent(request: ChatRequest):
    try:
        # Convert Pydantic history objects to dictionaries
        history_dicts = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg.dict()
            for msg in request.history
        ]
        
        response_text = await agent_service.arun_agent(
            message=request.message,
            history=history_dicts,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature
        )
        return ChatResponse(response=response_text, status="success")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
