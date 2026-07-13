import uuid
import logging
from datetime import datetime, timezone
from typing import Optional
from pymongo import ReturnDocument
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Depends, Query
from app.schemas.mongo_chat import (
    MongoChatRequest,
    MongoChatResponse,
    TokenUsage,
    ConversationListResponse,
    MessageListResponse
)
from app.services.agent import agent_service
from app.services.llm_factory import LLMFactory
from app.services.memory_service import memory_service
from app.core.config import settings
from app.core.auth import get_api_key
from app.core.rate_limit import limiter
from app.core.security import sanitize_mongodb_id, validate_dynamic_tools_metadata
from app.services.tool_safety_guard import tool_safety_guard

logger = logging.getLogger("app.api.mongo_chat")

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(get_api_key)])

@router.post(
    "/agent/mongo-query", 
    response_model=MongoChatResponse,
    summary="Truy vấn AI Agent lưu trữ lịch sử qua MongoDB (Primitive Memory)",
    description=(
        "Gửi câu hỏi tới AI Agent. Lịch sử hội thoại sẽ được quản lý và lưu trữ "
        "tự động trong MongoDB dưới dạng Primitive Memory, hỗ trợ cơ chế trượt (Sliding Window) "
        "và trả về báo cáo số token tiêu thụ thực tế để C# quản lý Quota."
    )
)
@limiter.limit("15/minute")
async def query_mongo_agent(request: Request, chat_req: MongoChatRequest, background_tasks: BackgroundTasks):
    # 0. Quét Prompt Injection trực tiếp trên tin nhắn của người dùng đầu vào
    if tool_safety_guard.check_prompt_injection(chat_req.message):
        raise HTTPException(
            status_code=400,
            detail="Yêu cầu bị từ chối do phát hiện dấu hiệu chỉ dẫn không an toàn (Prompt Injection)."
        )

    try:
        db = request.app.state.db
        if db is None:
            raise HTTPException(
                status_code=503, 
                detail="Dịch vụ cơ sở dữ liệu MongoDB hiện không khả dụng. Vui lòng kiểm tra lại kết nối cơ sở dữ liệu."
            )
        
        # 1. Giải quyết conversation_id
        conversation_id = chat_req.conversation_id
        if conversation_id:
            conversation_id = sanitize_mongodb_id(conversation_id, "conversation_id")
        conversation_exists = False

        # Chuẩn hóa dynamic_tools_metadata sớm để trích xuất resolved_workspace_id và resolved_user_id
        dynamic_tools_metadata = chat_req.dynamic_tools_metadata
        if not isinstance(dynamic_tools_metadata, dict):
            dynamic_tools_metadata = None
        else:
            dynamic_tools_metadata = validate_dynamic_tools_metadata(dynamic_tools_metadata)

        workspace_id_from_meta = None
        user_id_from_meta = None
        if dynamic_tools_metadata:
            workspace_id_from_meta = dynamic_tools_metadata.get("workspaceId") or dynamic_tools_metadata.get("workspace_id") or dynamic_tools_metadata.get("workspace_id")
            user_id_from_meta = dynamic_tools_metadata.get("userId") or dynamic_tools_metadata.get("user_id") or dynamic_tools_metadata.get("user_id")

        resolved_workspace_id = chat_req.workspace_id or (str(workspace_id_from_meta) if workspace_id_from_meta is not None else None)
        resolved_user_id = chat_req.user_id or (str(user_id_from_meta) if user_id_from_meta is not None else None)

        if resolved_workspace_id:
            resolved_workspace_id = sanitize_mongodb_id(resolved_workspace_id, "workspace_id")
        if resolved_user_id:
            resolved_user_id = sanitize_mongodb_id(resolved_user_id, "user_id")

        # Kiểm tra điều kiện bắt đầu hội thoại mới khi tắt ALLOW_ANONYMOUS_CHAT
        is_starting_new = (not conversation_id) or chat_req.force_new
        if is_starting_new and not settings.ALLOW_ANONYMOUS_CHAT and not resolved_user_id:
            raise HTTPException(
                status_code=403,
                detail="Hệ thống đang tắt chế độ ẩn danh. Cần có user_id để khởi tạo hội thoại mới."
            )

        if not conversation_id:
            if chat_req.force_new or not resolved_user_id:
                conversation_id = uuid.uuid4().hex
            else:
                try:
                    new_id = uuid.uuid4().hex
                    conv = await db["conversations"].find_one_and_update(
                        {"user_id": resolved_user_id},
                        {
                            "$setOnInsert": {
                                "conversation_id": new_id,
                                "user_id": resolved_user_id,
                                "title": "Hội thoại mới",
                                "created_at": datetime.now(timezone.utc),
                                "version": 1,
                                "metadata": {
                                    "provider": chat_req.provider or settings.LLM_PROVIDER,
                                    "model": chat_req.model or settings.LLM_MODEL,
                                    "temperature": chat_req.temperature if chat_req.temperature is not None else settings.LLM_TEMPERATURE
                                }
                            },
                            "$set": {
                                "updated_at": datetime.now(timezone.utc)
                            }
                        },
                        sort=[("updated_at", -1)],
                        upsert=True,
                        return_document=ReturnDocument.AFTER
                    )
                    conversation_id = conv["conversation_id"]
                    conversation_exists = True
                except Exception as db_exc:
                    logger.warning(f"Database conflict during atomic session resolve: {db_exc}. Falling back to new session.")
                    conversation_id = uuid.uuid4().hex
        else:
            conversation_exists = False

        # 1.5. Xác định provider của cuộc hội thoại và trạng thái tiêu đề
        conv_provider = chat_req.provider
        title_status = "generated"
        if db is not None:
            conv_doc = await db["conversations"].find_one({"conversation_id": conversation_id})
            if conv_doc:
                conversation_exists = True
                if not conv_provider and "metadata" in conv_doc:
                    conv_provider = conv_doc["metadata"].get("provider")
                if conv_doc.get("title", "") == "Hội thoại mới":
                    title_status = "generating"
            else:
                title_status = "generating"
        if not conv_provider:
            conv_provider = settings.LLM_PROVIDER

        # Lọc bỏ các tệp đính kèm giả lập (placeholder từ Swagger/C# default, e.g. "string")
        cleaned_attachments = []
        if chat_req.attachments:
            for att in chat_req.attachments:
                if att.file_id == "string" or att.file_name == "string" or att.storage_url == "string":
                    logger.info(f"Phát hiện và bỏ qua tệp đính kèm giả lập: {att.file_name}")
                    continue
                cleaned_attachments.append(att)

        # Xử lý tệp đính kèm (Ingest vào Document RAG) trước khi chạy agent
        if cleaned_attachments and db is not None:
            from app.services.document_rag import document_rag
            download_headers = None
            if dynamic_tools_metadata:
                download_headers = dynamic_tools_metadata.get("search_document_rag")
                
            clean_msg = chat_req.message.strip().lower()
            is_greeting_or_empty = (
                not clean_msg 
                or clean_msg in ("xin chào", "hi", "hello", "chào", "chào bạn", "gửi file", "tải file", "up file", "upload file")
                or len(clean_msg) < 5
            )
            
            if is_greeting_or_empty:
                # KỊCH BẢN 1: Tin nhắn trống/chào hỏi kèm file -> Chạy ngầm và trả về tức thời
                for att in cleaned_attachments:
                    background_tasks.add_task(
                        document_rag.ingest_attachment,
                        db=db,
                        file_id=att.file_id,
                        file_name=att.file_name,
                        file_type=att.file_type,
                        file_size=att.file_size,
                        storage_url=att.storage_url,
                        conversation_id=conversation_id,
                        user_id=resolved_user_id,
                        provider=conv_provider,
                        headers=download_headers
                    )
                
                # Đảm bảo conversation được tạo để lưu tin nhắn
                await memory_service.get_or_create_conversation(
                    db=db,
                    conversation_id=conversation_id,
                    user_id=resolved_user_id,
                    provider=chat_req.provider,
                    model=chat_req.model,
                    temperature=chat_req.temperature,
                    workspace_id=resolved_workspace_id
                )
                
                # Lưu tin nhắn User kèm file đính kèm
                from app.services.memory_service import count_tokens
                user_tokens = await count_tokens(chat_req.message, model_name=conv_provider)
                await memory_service.save_message(
                    db=db,
                    conversation_id=conversation_id,
                    role="user",
                    content=chat_req.message,
                    tokens_count=user_tokens,
                    attachments=cleaned_attachments,
                    metadata={"model_used": conv_provider}
                )
                
                # Lưu tin nhắn Assistant thông báo đang xử lý
                file_names_str = ", ".join([att.file_name for att in cleaned_attachments])
                response_text = f"Tài liệu '{file_names_str}' đang được phân tích trong nền. Bạn có thể bắt đầu đặt câu hỏi liên quan đến tài liệu này trong vài giây tới."
                assistant_tokens = await count_tokens(response_text, model_name=conv_provider)
                await memory_service.save_message(
                    db=db,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=response_text,
                    tokens_count=assistant_tokens,
                    metadata={"model_used": conv_provider}
                )
                
                # Tự động tạo tiêu đề nếu cuộc hội thoại mới
                if title_status == "generating":
                    background_tasks.add_task(
                        memory_service.trigger_title_generation_background,
                        request.app,
                        conversation_id,
                        chat_req.message or f"Tải lên file {file_names_str}",
                        LLMFactory.get_llm
                    )
                
                token_usage = TokenUsage(
                    prompt_tokens=user_tokens,
                    completion_tokens=assistant_tokens,
                    total_tokens=user_tokens + assistant_tokens
                )
                
                return MongoChatResponse(
                    response=response_text,
                    conversation_id=conversation_id,
                    status="processing",
                    usage=token_usage,
                    title_status=title_status
                )
            else:
                # KỊCH BẢN 2: Có câu hỏi chi tiết kèm file -> Chạy đồng bộ (sử dụng Batch Embedding rất nhanh)
                for att in cleaned_attachments:
                    try:
                        await document_rag.ingest_attachment(
                            db=db,
                            file_id=att.file_id,
                            file_name=att.file_name,
                            file_type=att.file_type,
                            file_size=att.file_size,
                            storage_url=att.storage_url,
                            conversation_id=conversation_id,
                            user_id=resolved_user_id,
                            provider=conv_provider,
                            headers=download_headers
                        )
                    except Exception as ingest_err:
                        logger.error(
                            f"Lỗi khi ingest file đính kèm đồng bộ {att.file_name} vào RAG: {ingest_err}. Tiếp tục cuộc hội thoại."
                        )

        # 2. Chạy agent với bộ nhớ MongoDB
        response_text, usage_data = await agent_service.arun_mongo_agent(
            db=db,
            conversation_id=conversation_id,
            user_id=resolved_user_id,
            message=chat_req.message,
            attachments=cleaned_attachments,
            provider=chat_req.provider,
            model=chat_req.model,
            temperature=chat_req.temperature,
            conversation_exists=conversation_exists,
            dynamic_tools_metadata=dynamic_tools_metadata,
            workspace_id=resolved_workspace_id
        )
        
        # 3. Đăng ký background task để kiểm tra và nén lịch sử hội thoại nếu cần thiết (Sliding Window)
        background_tasks.add_task(
            memory_service.trigger_summarization_background,
            request.app,
            conversation_id,
            LLMFactory.get_llm
        )
        
        if title_status == "generating":
            background_tasks.add_task(
                memory_service.trigger_title_generation_background,
                request.app,
                conversation_id,
                chat_req.message,
                LLMFactory.get_llm
            )
        
        token_usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        return MongoChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            status="success",
            usage=token_usage,
            title_status=title_status
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        raise exc

