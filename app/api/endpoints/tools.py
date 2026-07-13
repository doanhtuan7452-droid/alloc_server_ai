import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.schemas.tools import RegisterToolsPayload
from app.core.auth import verify_internal_access
from app.services.tool_manager import tool_manager
from app.services.tool_rag import tool_rag
from app.core.config import settings

logger = logging.getLogger("app.api.tools")

router = APIRouter(prefix="/tools", tags=["tools"])

@router.post(
    "/register",
    summary="Đăng ký và đồng bộ danh sách tool động từ server C#",
    description=(
        "Nhận danh sách cấu hình tool từ C#, sinh embedding cho mô tả tool, "
        "thực hiện ghi đè khử trùng lặp và đồng bộ toàn phần (xóa orphan tools). "
        "Yêu cầu xác thực nội bộ qua header X-Internal-Token hoặc X-Internal-Secret."
    )
)
async def register_tools(
    request: Request,
    payload: RegisterToolsPayload,
    _auth: str = Depends(verify_internal_access)
):
    db = request.app.state.db
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Kết nối cơ sở dữ liệu MongoDB hiện không khả dụng. Không thể đăng ký tool."
        )

    # 1. Sinh embeddings cho mô tả của từng tool trước khi lưu
    provider = settings.LLM_PROVIDER
    payload_names = []
    tool_documents = []

    for tool in payload.tools:
        payload_names.append(tool.name)
        
        # Sinh embedding cho mô tả của công cụ
        logger.info(f"Sinh embedding cho description của dynamic tool: {tool.name}")
        embedding = await tool_rag.get_embedding(tool.description, provider)
        
        doc = {
            "name": tool.name,
            "description": tool.description,
            "endpoint_url": tool.endpoint_url,
            "method": tool.method,
            "parameters": tool.parameters,
            "headers": tool.headers,
            "bundle_group": tool.bundle_group,
            "embedding": embedding,
            "updated_at": datetime.now(timezone.utc)
        }
        tool_documents.append(doc)

    # 2. Thực thi ghi dữ liệu nguyên tử (Transaction) kèm Fallback cho standalone MongoDB
    db_client = db.client
    
    async def perform_sync_operations(session=None):
        # A. Upsert từng tool trong danh sách đăng ký
        for doc in tool_documents:
            await db["external_tools"].update_one(
                {"name": doc["name"]},
                {"$set": doc},
                upsert=True,
                session=session
            )
        
        # B. Dọn dẹp Orphan Tools (xóa các tool có trong DB nhưng không có trong payload gửi từ C#)
        delete_result = await db["external_tools"].delete_many(
            {"name": {"$nin": payload_names}},
            session=session
        )
        logger.info(f"Đã xóa {delete_result.deleted_count} orphan tools khỏi database.")

        # C. Cập nhật tool_metadata timestamp để invalidate cache ở các FastAPI workers khác
        await db["tool_metadata"].update_one(
            {"id": "registry_metadata"},
            {
                "$set": {
                    "last_updated": datetime.now(timezone.utc)
                }
            },
            upsert=True,
            session=session
        )

    try:
        # Thử khởi chạy MongoDB Transaction
        async with await db_client.start_session() as session:
            session.start_transaction()
            try:
                await perform_sync_operations(session)
                await session.commit_transaction()
            except Exception as tx_exc:
                await session.abort_transaction()
                raise tx_exc
        logger.info("Đăng ký dynamic tools thành công thông qua MongoDB Transaction.")
    except Exception as exc:
        err_msg = str(exc)
        # Nếu lỗi do MongoDB là Standalone (không hỗ trợ Transaction)
        if "transaction numbers are only allowed on a replica set" in err_msg.lower() or "sessions are not supported" in err_msg.lower():
            logger.warning("MongoDB hiện tại không chạy dạng Replica Set. Thực hiện Fallback chạy tuần tự không transaction.")
            try:
                # Thực hiện đồng bộ tuần tự trực tiếp (không dùng session)
                await perform_sync_operations(session=None)
                logger.info("Đăng ký dynamic tools tuần tự thành công (Fallback mode).")
            except Exception as fallback_exc:
                logger.error(f"Lỗi khi thực hiện đăng ký tools tuần tự: {fallback_exc}")
                raise fallback_exc
        else:
            logger.error(f"Lỗi hệ thống khi đăng ký dynamic tools: {exc}")
            raise exc

    # 3. Làm mới cache RAM cục bộ của chính tiến trình FastAPI này ngay lập tức
    tool_manager.invalidate_local_cache()

    return {
        "status": "success",
        "message": f"Đồng bộ thành công {len(tool_documents)} tools động. Cache đã được invalidate.",
        "registered_tools": payload_names
    }

@router.get(
    "/active",
    summary="Liệt kê danh sách các công cụ hiện đang hoạt động và nạp trong hệ thống",
    description="Trả về danh sách tên và mô tả của tất cả các công cụ (Local và Dynamic)."
)
async def list_active_tools(request: Request, provider: Optional[str] = None):
    db = request.app.state.db
    prov = provider or settings.LLM_PROVIDER
    
    # Sử dụng get_filtered_tools với query rỗng để lấy toàn bộ danh sách tools (bỏ qua lọc tương đồng)
    # Vì query rỗng, ToolManager sẽ fallback nạp toàn bộ
    tools = await tool_manager.get_filtered_tools(
        query="",
        last_response=None,
        db=db,
        provider=prov
    )
    
    output = []
    for t in tools:
        output.append({
            "name": t.name,
            "description": t.description,
            "args_schema": t.args_schema.model_json_schema() if t.args_schema else None
        })
        
    return {
        "provider": prov,
        "total_active_tools": len(output),
        "tools": output
    }
