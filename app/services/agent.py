from typing import List, Optional
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.services.llm_factory import LLMFactory
from app.services.agent_tools import agent_tools
from app.core.config import settings

class AgentService:
    def __init__(self):
        self.system_prompt = (
            "Bạn là một trợ lý AI là Cố vấn Nhân sự và Quản lý Dự án cao cấp.\n"
            "Nhiệm vụ của bạn là hỗ trợ người dùng trò chuyện, phân bổ nhân sự vào công việc "
            "và đánh giá mức độ rủi ro của dự án bằng cách sử dụng các công cụ dự báo chuyên dụng.\n"
            "Hãy luôn sử dụng công cụ thích hợp khi người dùng hỏi các câu hỏi cần dự báo "
            "về sự phù hợp của nhân viên hoặc rủi ro của dự án. Nhớ giải thích kết quả một cách "
            "thân thiện, chuyên nghiệp, rõ ràng bằng tiếng Việt."
        )

    async def arun_agent(
        self,
        message: str,
        history: List[dict],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        # 1. Get the dynamic LLM instance via LLMFactory
        llm = LLMFactory.get_llm(provider=provider, model=model, temperature=temperature)

        # 2. Create the agent dynamically
        agent = create_agent(
            model=llm,
            tools=agent_tools,
            system_prompt=self.system_prompt
        )

        # 3. Format history and current message
        messages = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role in ("assistant", "agent"):
                messages.append(AIMessage(content=content))

        # Add current user query
        messages.append(HumanMessage(content=message))

        # 4. Invoke the agent graph asynchronously
        result = await agent.ainvoke({"messages": messages})

        # 5. Extract the last AI message
        out_messages = result.get("messages", [])
        if not out_messages:
            return "Xin lỗi, tôi không thể xử lý yêu cầu lúc này."
        
        last_msg = out_messages[-1]
        content = last_msg.content
        
        # Parse list-based content blocks (often returned by newer multimodal Gemini models)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            return "".join(text_parts)
            
        return str(content)

agent_service = AgentService()
