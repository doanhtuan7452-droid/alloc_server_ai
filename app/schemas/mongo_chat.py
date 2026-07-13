from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime

class AttachmentSchema(BaseModel):
    file_id: str
    file_name: str
    file_type: str
    file_size: int
    storage_url: str
    extracted_text: Optional[str] = None

class TokenUsage(BaseModel):
    prompt_tokens: int = Field(0, description="Token đầu vào")
    completion_tokens: int = Field(0, description="Token đầu ra")
    total_tokens: int = Field(0, description="Tổng token tiêu thụ")

class MongoChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="ID phiên hội thoại đồng bộ từ C# (tùy chọn)")
    user_id: Optional[str] = Field(None, description="ID người dùng gửi tin nhắn")
    workspace_id: Optional[str] = Field(None, alias="workspaceId", description="ID workspace tương ứng")
    message: str = Field(..., description="Tin nhắn hiện tại của người dùng")
    attachments: Optional[List[AttachmentSchema]] = Field(default=None, description="Danh sách file đính kèm")
    provider: Optional[str] = Field(None, description="Đè cấu hình Provider LLM")
    model: Optional[str] = Field(None, description="Đè cấu hình Model LLM")
    temperature: Optional[float] = Field(None, description="Đè cấu hình Temperature")
    force_new: Optional[bool] = Field(False, description="Bắt buộc khởi tạo phiên hội thoại mới")
    dynamic_tools_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Map chứa thông tin bối cảnh và header động gửi kèm cho từng tool_name")

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)


class MongoChatResponse(BaseModel):
    response: str = Field(..., description="Nội dung phản hồi từ AI Agent")
    conversation_id: str = Field(..., description="ID phiên hội thoại tương ứng")
    status: str = Field(default="success")
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Báo cáo số token tiêu thụ cho C#")
    title_status: str = Field(default="generated", description="Trạng thái tiêu đề hội thoại (generating: đang tạo ngầm, generated: đã được tạo/tùy chỉnh)")

class ConversationResponse(BaseModel):
    id: Optional[str] = Field(None, description="MongoDB ObjectId as string")
    conversation_id: str = Field(..., description="ID phiên hội thoại")
    user_id: Optional[str] = Field(None, description="ID người dùng sở hữu")
    title: str = Field(..., description="Tiêu đề hội thoại")
    created_at: datetime = Field(..., description="Thời gian tạo")
    updated_at: datetime = Field(..., description="Thời gian cập nhật cuối")
    summary: str = Field("", description="Tóm tắt nội dung hội thoại")
    version: int = Field(1, description="Phiên bản OCC")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Cấu hình LLM của hội thoại")

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

class ConversationListResponse(BaseModel):
    total: int = Field(..., description="Tổng số cuộc hội thoại")
    conversations: List[ConversationResponse] = Field(..., description="Danh sách các cuộc hội thoại")

class MessageResponse(BaseModel):
    id: Optional[str] = Field(None, description="MongoDB ObjectId as string")
    conversation_id: str = Field(..., description="ID phiên hội thoại")
    role: str = Field(..., description="Vai trò gửi tin nhắn")
    content: str = Field(..., description="Nội dung tin nhắn")
    timestamp: datetime = Field(..., description="Thời gian gửi")
    tokens_count: int = Field(0, description="Số lượng token tiêu thụ")
    attachments: Optional[List[AttachmentSchema]] = Field(None, description="Tệp đính kèm")
    rag_sources: Optional[List[Dict[str, Any]]] = Field(None, description="Nguồn tài liệu RAG")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata bổ sung")

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

class MessageListResponse(BaseModel):
    conversation_id: str = Field(..., description="ID phiên hội thoại")
    title: str = Field(..., description="Tiêu đề cuộc hội thoại")
    summary: str = Field(..., description="Tóm tắt ngữ cảnh cuộc hội thoại")
    total: int = Field(..., description="Tổng số tin nhắn trong cuộc hội thoại")
    messages: List[MessageResponse] = Field(..., description="Danh sách chi tiết tin nhắn")
