from datetime import datetime, timezone
import logging
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, Request
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.config import settings
from app.schemas.mongo_chat import AttachmentSchema, TokenUsage

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger("app.memory")

# Fallback token counting
async def count_tokens(text: str, llm: Any = None, model_name: str = "gpt-4o-mini") -> int:
    """Đếm số lượng token chính xác offline bằng tiktoken"""
    if not text:
        return 0
            
    # OpenAI & Tiktoken offline counting
    if tiktoken:
        try:
            # Chọn encoding phù hợp cho dòng model gpt-4o / gpt-4o-mini
            if "gpt-4o" in model_name:
                encoding = tiktoken.get_encoding("o200k_base")
            else:
                encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except Exception:
            try:
                # Bảng mã mặc định của các đời GPT-3.5/4 trước đó
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:
                pass
                
    # Ước lượng thô nếu không có tiktoken (khoảng 4 ký tự / token)
    return len(text) // 4

class MemoryService:
    def get_db(self, request_or_app: Any) -> Any:
        """Helper lấy db client từ app state"""
        if hasattr(request_or_app, "state") and hasattr(request_or_app.state, "db"):
            return request_or_app.state.db
        if hasattr(request_or_app, "app") and hasattr(request_or_app.app, "state") and hasattr(request_or_app.app.state, "db"):
            return request_or_app.app.state.db
        # Fallback trong trường hợp chạy test hoặc context độc lập
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        return client[settings.MONGODB_DB_NAME]

    async def get_or_create_conversation(self, db: Any, conversation_id: str, user_id: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None, workspace_id: Optional[str] = None) -> Dict[str, Any]:
        """Lấy thông tin hội thoại hoặc khởi tạo nếu chưa tồn tại"""
        conv = await db["conversations"].find_one({"conversation_id": conversation_id})
        if not conv:
            conv = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "workspace_id": workspace_id,
                "title": "Hội thoại mới",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "summary": "",
                "version": 1,
                "metadata": {
                    "provider": provider or settings.LLM_PROVIDER,
                    "model": model or settings.LLM_MODEL,
                    "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE
                }
            }
            await db["conversations"].insert_one(conv)
        return conv


    async def get_history(self, db: Any, conversation_id: str, limit: int = 6) -> Tuple[List[Dict[str, Any]], str]:
        """Lấy N tin nhắn gần nhất và tóm tắt ngữ cảnh cũ của hội thoại (mặc định limit=6)"""
        # 1. Lấy thông tin tóm tắt hội thoại
        conv = await db["conversations"].find_one({"conversation_id": conversation_id})
        summary = conv.get("summary", "") if conv else ""
        
        # 2. Lấy danh sách tin nhắn gần nhất
        messages_cursor = db["messages"].find({"conversation_id": conversation_id}).sort("timestamp", -1).limit(limit)
        db_messages = await messages_cursor.to_list(length=limit)
        
        # Đảo chiều để hiển thị theo thứ tự thời gian tăng dần
        db_messages.reverse()
        
        formatted_history = []
        for msg in db_messages:
            formatted_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        return formatted_history, summary

    async def save_message(
        self,
        db: Any,
        conversation_id: str,
        role: str,
        content: str,
        tokens_count: int,
        attachments: Optional[List[AttachmentSchema]] = None,
        rag_sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Lưu tin nhắn mới vào MongoDB và cập nhật updated_at của conversation"""
        attachments_data = []
        if attachments:
            for att in attachments:
                attachments_data.append(att.model_dump() if hasattr(att, "model_dump") else att)

        msg_doc = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc),
            "tokens_count": tokens_count,
            "attachments": attachments_data,
            "rag_sources": rag_sources or [],
            "metadata": metadata or {}
        }
        
        await db["messages"].insert_one(msg_doc)
        
        # Cập nhật thời gian tương tác cuối cùng của cuộc hội thoại
        await db["conversations"].update_one(
            {"conversation_id": conversation_id},
            {"$set": {"updated_at": datetime.now(timezone.utc)}}
        )

    async def trigger_summarization_background(self, app: FastAPI, conversation_id: str, llm_factory_func: Any) -> None:
        """Hàm kích hoạt tóm tắt lịch sử hội thoại chạy ngầm (FastAPI BackgroundTask)"""
        db = self.get_db(app)
        
        # 1. Lấy cấu hình hội thoại
        conv = await db["conversations"].find_one({"conversation_id": conversation_id})
        if not conv:
            return
            
        # 2. Đếm tổng số tin nhắn
        msg_count = await db["messages"].count_documents({"conversation_id": conversation_id})
        
        # Ngưỡng kích hoạt tóm tắt: Nếu tổng số tin nhắn vượt quá 10 tin nhắn
        # Ta giữ lại 6 tin nhắn mới nhất, phần còn lại sẽ được nén vào summary
        trigger_limit = 12
        keep_limit = 6
        
        if msg_count <= trigger_limit:
            return
            
        logger.info(f"Kích hoạt Sliding Window Summarization cho {conversation_id}. Tổng tin nhắn: {msg_count}")
        
        # Lấy tất cả tin nhắn ngoại trừ 6 tin nhắn mới nhất
        old_messages_cursor = db["messages"].find({"conversation_id": conversation_id}).sort("timestamp", 1).limit(msg_count - keep_limit)
        old_messages = await old_messages_cursor.to_list(length=(msg_count - keep_limit))
        
        if not old_messages:
            return
            
        # Định dạng tin nhắn để gửi cho LLM tóm tắt
        history_text_lines = []
        for m in old_messages:
            role_label = "Người dùng" if m["role"] == "user" else "Trợ lý AI"
            history_text_lines.append(f"{role_label}: {m['content']}")
        new_history_text = "\n".join(history_text_lines)
        
        # Lấy LLM để thực hiện tóm tắt ngữ cảnh
        metadata_conf = conv.get("metadata", {})
        provider = metadata_conf.get("provider", settings.LLM_PROVIDER)
        model_name = metadata_conf.get("model", settings.LLM_MODEL)
        
        llm = llm_factory_func(provider=provider, model=model_name, temperature=0.3)
        old_summary = conv.get("summary", "")
        
        prompt = (
            "Bạn là trợ lý AI chuyên nghiệp. Nhiệm vụ của bạn là tổng hợp các tin nhắn hội thoại cũ "
            "và tóm tắt lịch sử hiện tại thành một đoạn tóm tắt ngữ cảnh mới, ngắn gọn, súc tích bằng tiếng Việt.\n"
            "Hãy lưu ý giữ lại các thông tin cốt lõi như: tên dự án, tên nhân viên, các yêu cầu đặc biệt của người dùng.\n\n"
            f"Tóm tắt cũ (nếu có):\n{old_summary}\n\n"
            f"Lịch sử hội thoại mới bổ sung:\n{new_history_text}\n\n"
            "Hãy viết đoạn tóm tắt mới gọn gàng (không quá 200 từ) và trực tiếp, không thêm kính ngữ hay giới thiệu."
        )
        
        try:
            # Gọi LLM sinh tóm tắt mới (chỉ chạy duy nhất 1 lần ở ngoài vòng lặp OCC)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            new_summary = response.content.strip()
        except Exception as e:
            logger.error(f"Lỗi trong quá trình tạo tóm tắt hội thoại bằng LLM: {str(e)}")
            return

        # Thực hiện OCC tối đa 3 lần thử lại chỉ để cập nhật DB
        for attempt in range(1, 4):
            conv = await db["conversations"].find_one({"conversation_id": conversation_id})
            if not conv:
                return
                
            current_version = conv.get("version", 1)
            
            try:
                # Cập nhật bằng cơ chế Optimistic Concurrency Control (OCC)
                result = await db["conversations"].update_one(
                    {"conversation_id": conversation_id, "version": current_version},
                    {
                        "$set": {"summary": new_summary, "updated_at": datetime.now(timezone.utc)},
                        "$inc": {"version": 1}
                    }
                )
                
                if result.modified_count > 0:
                    logger.info(f"Cập nhật summary thành công cho {conversation_id} (Version: {current_version} -> {current_version+1})")
                    break
                else:
                    logger.warning(f"OCC Retry Attempt {attempt}/3 thất bại do dữ liệu bị sửa đổi song song.")
            except Exception as e:
                logger.error(f"Lỗi khi lưu DB summary: {str(e)}")
                break
        else:
            logger.error(f"OCC Retry hoàn toàn thất bại sau 3 lần thử cho hội thoại {conversation_id}. Giữ nguyên summary cũ.")

    async def trigger_title_generation_background(
        self,
        app: FastAPI,
        conversation_id: str,
        first_message: str,
        llm_factory_func: Any
    ) -> None:
        """Sinh tiêu đề tự động chạy ngầm dựa trên nội dung tin nhắn đầu tiên sử dụng LLM và OCC"""
        db = self.get_db(app)
        
        conv = await db["conversations"].find_one({"conversation_id": conversation_id})
        if not conv:
            logger.warning(f"Không tìm thấy phiên hội thoại {conversation_id} để đặt tiêu đề.")
            return
            
        # Chỉ sinh tiêu đề nếu tiêu đề hiện tại đang là "Hội thoại mới"
        current_title = conv.get("title", "")
        if current_title != "Hội thoại mới":
            logger.info(f"Tiêu đề đã được thay đổi thành '{current_title}'. Bỏ qua tự động sinh tiêu đề.")
            return
            
        # 1. Gọi LLM sinh tiêu đề (chỉ chạy duy nhất 1 lần ở ngoài vòng lặp OCC)
        metadata_conf = conv.get("metadata", {})
        provider = metadata_conf.get("provider", settings.LLM_PROVIDER)
        model_name = metadata_conf.get("model", settings.LLM_MODEL)
        
        llm = llm_factory_func(provider=provider, model=model_name, temperature=0.3)
        
        prompt = (
            "Bạn là một trợ lý AI chuyên nghiệp. Hãy đọc tin nhắn của người dùng dưới đây "
            "và tạo ra một tiêu đề hội thoại ngắn gọn (tối đa 5-7 từ) bằng tiếng Việt mô tả nội dung cuộc trò chuyện.\n"
            "RẤT QUAN TRỌNG: Không thêm bất kỳ lời giải thích nào, không dùng tiền tố, không sử dụng dấu ngoặc kép hoặc ngoặc đơn, "
            "chỉ trả về chuỗi tiêu đề trực tiếp.\n"
            "Nếu tin nhắn quá ngắn hoặc chào hỏi chung chung (như 'xin chào', 'hi', 'chào bạn'), hãy đặt tiêu đề là 'Trò chuyện chung'.\n\n"
            f"Tin nhắn: {first_message}"
        )
        
        new_title = ""
        try:
            # Gọi LLM sinh tiêu đề
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            new_title = response.content.strip()
            new_title = new_title.strip('"\'')
        except Exception as e:
            logger.warning(f"Lỗi khi gọi LLM sinh tiêu đề: {e}. Sử dụng logic fallback.")
            import re
            clean_msg = re.sub(r'[^\w\s\d]', '', first_message).strip()
            words = clean_msg.split()
            if words:
                new_title = " ".join(words[:6])
            else:
                new_title = "Trò chuyện chung"
                
        if not new_title:
            new_title = "Trò chuyện chung"

        # 2. Thực hiện OCC tối đa 3 lần thử lại chỉ để cập nhật DB
        for attempt in range(1, 4):
            conv = await db["conversations"].find_one({"conversation_id": conversation_id})
            if not conv:
                return
                
            # Chỉ sinh tiêu đề nếu tiêu đề vẫn là "Hội thoại mới"
            if conv.get("title", "") != "Hội thoại mới":
                logger.info(f"Tiêu đề đã được thay đổi song song. Bỏ qua.")
                break
                
            current_version = conv.get("version", 1)
            
            try:
                result = await db["conversations"].update_one(
                    {"conversation_id": conversation_id, "version": current_version},
                    {
                        "$set": {"title": new_title, "updated_at": datetime.now(timezone.utc)},
                        "$inc": {"version": 1}
                    }
                )
                
                if result.modified_count > 0:
                    logger.info(f"Tự động đặt tiêu đề '{new_title}' thành công cho {conversation_id} (Version: {current_version} -> {current_version+1})")
                    break
                else:
                    logger.warning(f"OCC Title Retry Attempt {attempt}/3 thất bại do dữ liệu bị sửa đổi song song.")
            except Exception as e:
                logger.error(f"Lỗi khi update database title: {str(e)}")
                break
        else:
            logger.error(f"OCC Title Retry hoàn toàn thất bại sau 3 lần thử cho hội thoại {conversation_id}.")

    def _serialize_mongo_doc(self, doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Chuyển đổi ObjectId thành chuỗi để tránh lỗi JSON serialization"""
        if not doc:
            return doc
        new_doc = doc.copy()
        if "_id" in new_doc:
            new_doc["id"] = str(new_doc.pop("_id"))
        return new_doc

    async def get_conversations(
        self,
        db: Any,
        user_id: Optional[str] = None,
        limit: int = 20,
        skip: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Lấy danh sách cuộc hội thoại phân trang, có thể lọc theo user_id"""
        query = {}
        if user_id:
            query["user_id"] = user_id
            
        total = await db["conversations"].count_documents(query)
        cursor = db["conversations"].find(query).sort("updated_at", -1).skip(skip).limit(limit)
        db_conversations = await cursor.to_list(length=limit)
        
        conversations = [self._serialize_mongo_doc(c) for c in db_conversations]
        return conversations, total

    async def get_conversation_messages(
        self,
        db: Any,
        conversation_id: str,
        limit: int = 50,
        skip: int = 0,
        order: str = "asc"
    ) -> Tuple[List[Dict[str, Any]], int, Optional[Dict[str, Any]]]:
        """Lấy danh sách tin nhắn của cuộc hội thoại phân trang"""
        # 1. Lấy thông tin hội thoại
        conv = await db["conversations"].find_one({"conversation_id": conversation_id})
        if not conv:
            return [], 0, None
            
        # 2. Đếm tổng số tin nhắn
        total = await db["messages"].count_documents({"conversation_id": conversation_id})
        
        # 3. Lấy tin nhắn
        sort_dir = 1 if order == "asc" else -1
        cursor = db["messages"].find({"conversation_id": conversation_id}).sort("timestamp", sort_dir).skip(skip).limit(limit)
        db_messages = await cursor.to_list(length=limit)
        
        messages = [self._serialize_mongo_doc(m) for m in db_messages]
        return messages, total, self._serialize_mongo_doc(conv)

memory_service = MemoryService()
