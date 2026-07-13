import logging
from typing import Optional, Dict, Any
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Request, Depends, Query
from fastapi.responses import ORJSONResponse

from app.services.memory_service import memory_service
from app.core.auth import verify_internal_access
from app.schemas.mongo_chat import ConversationListResponse, MessageListResponse
from app.core.security import sanitize_mongodb_id

logger = logging.getLogger("app.api.internal_chat")

router = APIRouter(
    prefix="/chat", 
    tags=["internal_chat"], 
    dependencies=[Depends(verify_internal_access)]
)

def clean_doc(doc: Any) -> Any:
    """Đảm bảo chuyển đổi tất cả bson.ObjectId thành str đệ quy để orjson không bị lỗi"""
    if doc is None:
        return None
    if isinstance(doc, ObjectId):
        return str(doc)
    if isinstance(doc, dict):
        cleaned = {}
        for k, v in doc.items():
            if k == "_id":
                cleaned["id"] = str(v)
            else:
                cleaned[k] = clean_doc(v)
        return cleaned
    if isinstance(doc, list):
        return [clean_doc(item) for item in doc]
    return doc

@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    response_class=ORJSONResponse,
    summary="Lấy danh sách các cuộc hội thoại (Internal API)",
    description=(
        "Lấy danh sách cuộc hội thoại lưu trữ trong MongoDB. "
        "Yêu cầu xác thực Internal Secret và IP Whitelisting. Bỏ qua bước kiểm tra chậm của Pydantic."
    )
)
async def list_conversations(
    request: Request,
    user_id: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100, description="Số lượng cuộc hội thoại tối đa trả về"),
    skip: int = Query(default=0, ge=0, le=1000, description="Số lượng cuộc hội thoại bỏ qua")
):
    db = request.app.state.db
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Dịch vụ cơ sở dữ liệu MongoDB hiện không khả dụng."
        )
    try:
        # Sanitize user_id if provided
        sanitized_user_id = user_id
        if user_id:
            sanitized_user_id = sanitize_mongodb_id(user_id, "user_id")

        conversations, total = await memory_service.get_conversations(
            db=db, user_id=sanitized_user_id, limit=limit, skip=skip
        )
        # Sử dụng clean_doc để lọc triệt để ObjectId, trả về ORJSONResponse
        cleaned_conversations = clean_doc(conversations)
        return ORJSONResponse(
            content={
                "total": total, 
                "conversations": cleaned_conversations
            }
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        raise exc


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=MessageListResponse,
    response_class=ORJSONResponse,
    summary="Lấy danh sách tin nhắn trong cuộc hội thoại (Internal API)",
    description=(
        "Lấy chi tiết tin nhắn của cuộc hội thoại có kiểm định IDOR qua user_id. "
        "Yêu cầu xác thực Internal Secret và IP Whitelisting. Phản hồi nhanh qua orjson."
    )
)
async def list_conversation_messages(
    request: Request,
    conversation_id: str,
    user_id: str = Query(..., description="ID người dùng sở hữu để kiểm định IDOR"),
    limit: int = Query(default=50, ge=1, le=100, description="Số lượng tin nhắn tối đa trả về"),
    skip: int = Query(default=0, ge=0, le=1000, description="Số lượng tin nhắn bỏ qua"),
    order: str = Query(default="asc", regex="^(asc|desc)$", description="Thứ tự sắp xếp tin nhắn (asc hoặc desc)")
):
    db = request.app.state.db
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Dịch vụ cơ sở dữ liệu MongoDB hiện không khả dụng."
        )
    try:
        # Sanitize IDs to mitigate NoSQL injection
        sanitized_conversation_id = sanitize_mongodb_id(conversation_id, "conversation_id")
        sanitized_user_id = sanitize_mongodb_id(user_id, "user_id")

        # 1. Kiểm định IDOR: Truy vấn cuộc hội thoại và kẹp chéo điều kiện user_id để xác thực sở hữu
        conv = await db["conversations"].find_one({
            "conversation_id": sanitized_conversation_id,
            "user_id": sanitized_user_id
        })
        if not conv:
            # Security by Obscurity: Trả về 404 Not Found để che giấu sự tồn tại
            raise HTTPException(
                status_code=404,
                detail="Not Found"
            )
            
        # 2. Lấy dữ liệu tin nhắn nếu vượt qua kiểm tra IDOR
        messages, total, serialized_conv = await memory_service.get_conversation_messages(
            db=db, conversation_id=sanitized_conversation_id, limit=limit, skip=skip, order=order
        )
        
        # Làm sạch ObjectId
        cleaned_messages = clean_doc(messages)
        cleaned_conv = clean_doc(serialized_conv)
        
        return ORJSONResponse(
            content={
                "conversation_id": sanitized_conversation_id,
                "title": cleaned_conv.get("title", "Hội thoại mới"),
                "summary": cleaned_conv.get("summary", ""),
                "total": total,
                "messages": cleaned_messages
            }
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        raise exc
