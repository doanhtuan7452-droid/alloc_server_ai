from pydantic import BaseModel, Field, AliasChoices
from typing import List, Optional, Dict, Any

class ToolInfo(BaseModel):
    name: str = Field(..., description="Tên duy nhất của công cụ, dùng làm định danh (Unique ID)")
    description: str = Field(..., description="Mô tả chi tiết chức năng của công cụ để LLM và RAG nhận diện")
    endpoint_url: str = Field(
        ..., 
        validation_alias=AliasChoices("endpoint_url", "endpointUrl"), 
        description="URL của API bên ngoài để thực thi cuộc gọi proxy"
    )
    method: str = Field("POST", description="Phương thức HTTP (GET, POST, PUT, DELETE, v.v.)")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Cấu trúc JSON Schema mô tả các tham số đầu vào")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Các HTTP Headers tĩnh gửi kèm theo cuộc gọi (ví dụ: Static Auth Key)")
    bundle_group: Optional[str] = Field(
        default=None, 
        validation_alias=AliasChoices("bundle_group", "bundleGroup"), 
        description="Tên nhóm các công cụ có liên quan mật thiết để nạp đồng thời (tối đa 5)"
    )

class RegisterToolsPayload(BaseModel):
    tools: List[ToolInfo] = Field(..., description="Danh sách các công cụ động gửi từ C# để đăng ký/đồng bộ")
