import httpx
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pydantic import create_model, BaseModel, Field
from langchain_core.tools import StructuredTool, BaseTool

from app.core.config import settings
from app.services.agent_tools import agent_tools
from app.services.search_tool import internet_search_tool
from app.services.tool_rag import tool_rag

logger = logging.getLogger("app.services.tool_manager")

class EmptyModel(BaseModel):
    pass

class ToolManager:
    def __init__(self):
        # RAM Cache cho các dynamic tools tải từ DB
        self._cached_dynamic_tool_docs: List[dict] = []
        self._last_synced_timestamp: Optional[datetime] = None

    def invalidate_local_cache(self):
        """Buộc worker xóa bộ đệm RAM để tải lại từ DB ở lượt tiếp theo."""
        self._last_synced_timestamp = None
        self._cached_dynamic_tool_docs = []
        logger.info("RAM vector cache của Tool Manager đã được làm mới (Invalidated).")

    async def _sync_cache_from_db(self, db: Any) -> List[dict]:
        """Đồng bộ danh sách tool động từ DB dựa trên Version-tracked Cache."""
        if db is None:
            return []
            
        try:
            # Đọc tài liệu metadata phiên bản duy nhất
            meta = await db["tool_metadata"].find_one({"id": "registry_metadata"})
            db_last_updated = meta.get("last_updated") if meta else None
            
            # Nếu chưa có timestamp hoặc timestamp trong DB lớn hơn timestamp cục bộ, đồng bộ lại
            is_stale = (
                self._last_synced_timestamp is None 
                or db_last_updated is None 
                or db_last_updated > self._last_synced_timestamp
            )
            
            if is_stale:
                logger.info("Phát hiện cache cũ hoặc thay đổi từ DB. Đang tải lại danh sách dynamic tools...")
                cursor = db["external_tools"].find({})
                docs = await cursor.to_list(length=100)
                
                self._cached_dynamic_tool_docs = docs
                self._last_synced_timestamp = db_last_updated or datetime.now(timezone.utc)
                logger.info(f"Đã nạp {len(docs)} dynamic tools vào RAM cache. Sync timestamp: {self._last_synced_timestamp}")
                
            return self._cached_dynamic_tool_docs
        except Exception as e:
            logger.error(f"Lỗi khi đồng bộ cache dynamic tools từ MongoDB: {e}")
            return self._cached_dynamic_tool_docs

    def _create_pydantic_model(self, tool_name: str, parameters: Optional[dict]) -> type:
        """Tạo động Pydantic Model từ JSON Schema parameters."""
        if not parameters:
            return EmptyModel
            
        fields = {}
        # Hỗ trợ cấu trúc OpenAPI parameters hoặc JSON Schema properties
        properties = parameters.get("properties", parameters)
        required_fields = parameters.get("required", [])

        for name, info in properties.items():
            if not isinstance(info, dict):
                continue
            
            # Bỏ qua xác thực Pydantic cho các trường ID và chữ ký bị ép kiểu (Forceful Injection)
            if name.lower() in ("userid", "workspaceid", "contextsignature"):
                continue
                
            val_type = str
            t = info.get("type", "string").lower()
            if t in ("integer", "int"):
                val_type = int
            elif t in ("number", "float"):
                val_type = float
            elif t in ("boolean", "bool"):
                val_type = bool
            elif t == "array":
                val_type = list
            elif t == "object":
                val_type = dict

            desc = info.get("description", "")
            # Xác định xem trường có bắt buộc không
            is_required = name in required_fields or info.get("required", False)
            default = ... if is_required else info.get("default", None)
            
            fields[name] = (val_type, Field(default=default, description=desc))
            
        try:
            return create_model(f"DynamicArgs_{tool_name}", **fields)
        except Exception as e:
            logger.error(f"Lỗi khi tạo Pydantic model cho tool {tool_name}: {e}. Sử dụng EmptyModel làm dự phòng.")
            return EmptyModel

    def _build_request_scoped_tool(
        self, 
        doc: dict, 
        dynamic_tools_metadata: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None
    ) -> BaseTool:
        """
        Khởi tạo StructuredTool động dạng Request-Scoped Closure.
        Đảm bảo headers/credentials được cô lập tuyệt đối cho mỗi lượt request.
        """
        from app.core.security import validate_dynamic_tools_metadata
        dynamic_tools_metadata = validate_dynamic_tools_metadata(dynamic_tools_metadata)
        name = doc["name"]
        description = doc["description"]
        endpoint_url = doc["endpoint_url"]
        method = doc.get("method", "POST")
        static_headers = doc.get("headers") or {}
        parameters = doc.get("parameters")
        
        # Tạo Pydantic model động làm args_schema (loại trừ các trường ID đã được tiêm ẩn)
        ArgsModel = self._create_pydantic_model(name, parameters)

        # Lấy dynamic headers của riêng request này (nếu có)
        request_headers = {}
        if isinstance(dynamic_tools_metadata, dict) and name in dynamic_tools_metadata:
            request_headers = dynamic_tools_metadata[name] or {}

        # Định nghĩa closure function cho thực thi công cụ
        async def _execute_dynamic_tool(**kwargs) -> str:
            # 1. Forceful Injection: Ép chèn trực tiếp các ID từ bối cảnh xác thực để chống hallucination/prompt injection
            payload = kwargs.copy()
            properties = parameters.get("properties", parameters) if parameters else {}
            for param_name in properties.keys():
                param_name_lower = param_name.lower()
                if param_name_lower == "userid" and user_id:
                    payload[param_name] = int(user_id) if (isinstance(user_id, str) and user_id.isdigit()) else user_id
                elif param_name_lower == "workspaceid" and workspace_id:
                    payload[param_name] = int(workspace_id) if (isinstance(workspace_id, str) and workspace_id.isdigit()) else workspace_id

            # 1.5. Lấy và ép chèn contextSignature (Force Inject), kiểm tra an toàn Fail-Fast
            context_signature = None
            if isinstance(dynamic_tools_metadata, dict):
                context_signature = (
                    dynamic_tools_metadata.get("contextSignature") 
                    or dynamic_tools_metadata.get("context_signature") 
                    or dynamic_tools_metadata.get("contextsignature")
                )
            
            if not context_signature:
                logger.warning(f"[ToolManager] Chặn thực thi dynamic tool {name} do thiếu chữ ký xác thực context.")
                return "Lỗi bảo mật hệ thống: Không thể thực thi công cụ do thiếu chữ ký xác thực thông tin tài khoản (contextSignature) hợp lệ. Vui lòng thực hiện trò chuyện trên kênh chính thức có đăng nhập."
                
            payload["contextSignature"] = context_signature

            # 2. Quét an toàn Prompt Injection qua ToolSafetyGuard
            from app.services.tool_safety_guard import tool_safety_guard
            is_safe, safety_err = tool_safety_guard.validate_tool_call(name, payload)
            if not is_safe:
                return f"Lỗi an toàn: {safety_err}"

            # Gộp static headers từ DB và request headers của user
            headers_payload = {}
            for k, v in static_headers.items():
                headers_payload[k.lower()] = v
            for k, v in request_headers.items():
                headers_payload[k.lower()] = v

            # Tự động chèn X-Internal-Token để xác thực nội bộ nếu chưa có
            if "x-internal-token" not in headers_payload and settings.INTERNAL_API_SECRET:
                headers_payload["x-internal-token"] = settings.INTERNAL_API_SECRET

            logger.info(f"Đang thực thi dynamic tool {name} -> {method} {endpoint_url}")
            
            async with httpx.AsyncClient() as client:
                try:
                    if method.upper() == "GET":
                        # Với GET, truyền tham số dưới dạng query params
                        resp = await client.request(
                            method=method,
                            url=endpoint_url,
                            params=payload,
                            headers=headers_payload,
                            timeout=10.0
                        )
                    else:
                        # Với POST/PUT, truyền dưới dạng JSON body
                        resp = await client.request(
                            method=method,
                            url=endpoint_url,
                            json=payload,
                            headers=headers_payload,
                            timeout=10.0
                        )
                    
                    # 3. Quản lý Self-Correction Loop: Trả mã lỗi thô về cho Agent thay vì raise HTTP Exception
                    if resp.status_code >= 400:
                        return f"Lỗi từ hệ thống C# (Mã {resp.status_code}): {resp.text}"
                        
                    return resp.text
                except Exception as exc:
                    err_msg = f"Lỗi kết nối khi gọi tool động {name} tới {endpoint_url}: {exc}"
                    logger.error(err_msg)
                    return err_msg

        # Tạo StructuredTool của LangChain
        return StructuredTool.from_function(
            coroutine=_execute_dynamic_tool,
            name=name,
            description=description,
            args_schema=ArgsModel
        )

    async def get_filtered_tools(
        self, 
        query: str, 
        last_response: Optional[str], 
        db: Any, 
        provider: str,
        dynamic_tools_metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None
    ) -> List[BaseTool]:
        """
        Tổng hợp tất cả local tools và dynamic tools từ MongoDB, chạy lọc RAG ngữ cảnh
        để lấy danh sách tối giản, áp dụng Tool Bundling và trả về tập hợp công cụ an toàn.
        """
        # 1. Đăng ký tất cả các local tools có sẵn
        all_candidates: List[Tuple[BaseTool, Optional[dict]]] = []
        
        # Nạp local ML tools từ agent_tools.py
        for t in agent_tools:
            # Ghi nhận dưới dạng tuple (tool_object, raw_db_document_or_none)
            all_candidates.append((t, None))
            
        # Nạp local search tool
        all_candidates.append((internet_search_tool, None))

        # Nạp Document RAG tool động theo phiên hội thoại
        if conversation_id and db is not None:
            from app.services.document_rag import document_rag
            
            class DocumentRAGInput(BaseModel):
                query: str = Field(..., description="Từ khóa hoặc câu hỏi cần tra cứu trong tài liệu đính kèm")

            async def search_document_rag_func(query: str) -> str:
                """Tìm kiếm thông tin chi tiết từ tài liệu đính kèm của cuộc hội thoại hiện tại."""
                results = await document_rag.search_rag(
                    db=db,
                    query=query,
                    conversation_id=conversation_id,
                    provider=provider,
                    top_k=settings.RAG_MAX_RESULTS
                )
                if not results:
                    return f"Không tìm thấy thông tin phù hợp nào trong tài liệu đính kèm cho truy vấn: '{query}'"
                
                formatted_results = []
                for idx, item in enumerate(results, 1):
                    formatted_results.append(
                        f"Đoạn {idx} (Từ tài liệu): {item['text']}\n"
                    )
                return "\n".join(formatted_results)

            document_rag_tool = StructuredTool.from_function(
                coroutine=search_document_rag_func,
                name="search_document_rag",
                description=(
                    "Sử dụng công cụ này khi người dùng hỏi các câu hỏi cần tìm kiếm, tra cứu, trích xuất "
                    "hoặc tóm tắt thông tin nằm trong các tài liệu đính kèm (files/attachments) được gửi kèm trong cuộc hội thoại."
                ),
                args_schema=DocumentRAGInput
            )
            all_candidates.append((document_rag_tool, None))

        # 2. Đồng bộ và nạp các dynamic tools từ DB
        dynamic_docs = await self._sync_cache_from_db(db)
        for doc in dynamic_docs:
            # Khởi tạo dynamic tool trong request scope
            scoped_tool = self._build_request_scoped_tool(
                doc=doc, 
                dynamic_tools_metadata=dynamic_tools_metadata,
                user_id=user_id,
                workspace_id=workspace_id
            )
            all_candidates.append((scoped_tool, doc))

        # 3. Chạy Tool RAG để lọc độ tương đồng ngữ cảnh
        # Làm giàu câu hỏi bằng ngữ cảnh đa lượt
        enriched_query = tool_rag.enrich_query(query, last_response)
        
        try:
            query_vector = await tool_rag.get_embedding(enriched_query, provider)
        except Exception as e:
            logger.error(f"Không thể sinh embedding cho query. Trực tiếp nạp fallback tools (internet_search, search_document_rag): {e}")
            return [item[0] for item in all_candidates if item[0].name in ("internet_search", "search_document_rag")]

        if not query_vector:
            logger.warning("Không lấy được vector cho query. Trực tiếp nạp fallback tools (internet_search, search_document_rag).")
            return [item[0] for item in all_candidates if item[0].name in ("internet_search", "search_document_rag")]

        # Tính toán điểm cosine cho từng tool
        scored_tools = []
        for tool_obj, doc in all_candidates:
            # Lấy vector mô tả
            if doc and "embedding" in doc and doc["embedding"]:
                # Tool động đã được sinh embedding khi đăng ký
                desc_vector = doc["embedding"]
            else:
                # Tool tĩnh cục bộ, sinh và cache embedding thông minh kết hợp RAM và MongoDB
                desc_vector = await tool_rag.get_or_create_static_tool_embedding_db(
                    db,
                    tool_obj.name, 
                    tool_obj.description, 
                    provider
                )
            
            score = tool_rag.cosine_similarity(query_vector, desc_vector)
            scored_tools.append((tool_obj, doc, score))
            logger.info(f"Tool: {tool_obj.name} | Cosine Similarity Score: {score:.4f}")

        # Lọc các tool vượt qua ngưỡng threshold (luôn giữ lại internet_search và search_document_rag)
        selected_pairs = [
            (t, doc) for t, doc, score in scored_tools 
            if score >= settings.TOOL_RAG_THRESHOLD or t.name in ("internet_search", "search_document_rag")
        ]
        
        # Nếu không có tool nào vượt qua ngưỡng (trừ search tools), lấy Top K tốt nhất
        business_selected = [t for t, _ in selected_pairs if t.name not in ("internet_search", "search_document_rag")]
        if not business_selected:
            sorted_scored = sorted(scored_tools, key=lambda x: x[2], reverse=True)
            # Lấy tối đa TOP_K_TOOLS bao gồm cả search tools
            top_k_items = sorted_scored[:settings.TOOL_RAG_TOP_K]
            selected_pairs = [(t, doc) for t, doc, score in top_k_items]
            # Chắc chắn rằng các công cụ bắt buộc luôn có mặt
            existing_names = {t.name for t, _ in selected_pairs}
            for t, doc, _ in scored_tools:
                if t.name in ("internet_search", "search_document_rag") and t.name not in existing_names:
                    selected_pairs.append((t, doc))

        # 4. Cơ chế Tool Bundling (Nạp chéo nhóm liên quan)
        final_tools_map = {t.name: t for t, _ in selected_pairs}
        
        # Lưu vết đếm số lượng tool được nạp cho từng bundle_group để bảo vệ token cost (max 5)
        bundle_group_counts = {}
        for _, doc in selected_pairs:
            if doc and doc.get("bundle_group"):
                bg = doc["bundle_group"]
                bundle_group_counts[bg] = bundle_group_counts.get(bg, 0) + 1

        # Duyệt lại toàn bộ danh sách ứng viên để nạp bù các tool cùng group nếu group đó được chọn
        for tool_obj, doc in all_candidates:
            if doc and doc.get("bundle_group"):
                bg = doc["bundle_group"]
                # Nếu group này đã được kích hoạt bởi RAG và chưa vượt quá giới hạn cứng (5)
                if bg in bundle_group_counts and tool_obj.name not in final_tools_map:
                    if bundle_group_counts[bg] < 5:
                        final_tools_map[tool_obj.name] = tool_obj
                        bundle_group_counts[bg] += 1
                        logger.info(f"Tool Bundling: Nạp thêm '{tool_obj.name}' do cùng nhóm '{bg}'")

        return list(final_tools_map.values())

tool_manager = ToolManager()
