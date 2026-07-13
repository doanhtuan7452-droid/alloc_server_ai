import os
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.file_service import file_service
from app.services.markitdown_service import markitdown_service
from app.services.tool_rag import tool_rag

logger = logging.getLogger("app.services.document_rag")

class DocumentRAGService:
    async def ingest_attachment(
        self,
        db: Any,
        file_id: str,
        file_name: str,
        file_type: str,
        file_size: int,
        storage_url: str,
        conversation_id: str,
        user_id: Optional[str],
        provider: str,
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Quy trình xử lý tệp đính kèm: Tải xuống -> Chuyển đổi Markdown -> Phân mảnh -> Sinh Vector -> Lưu MongoDB.
        """
        if db is None:
            logger.warning("Không có kết nối cơ sở dữ liệu MongoDB. Bỏ qua Ingest RAG.")
            return

        # 1. Kiểm tra tài liệu đã được xử lý chưa để tránh xử lý trùng lặp
        existing_doc = await db["rag_documents"].find_one({"file_id": file_id})
        if existing_doc:
            logger.info(f"Tài liệu {file_id} đã tồn tại trong RAG. Bỏ qua bước Ingestion.")
            return

        temp_file_path = None
        try:
            # 2. Tải file về máy cục bộ bất đồng bộ
            temp_file_path = await file_service.download_file(file_id, storage_url, headers)

            # 3. Chuyển đổi tệp sang định dạng Markdown
            markdown_content = await markitdown_service.convert_file_to_markdown(temp_file_path)

            if not markdown_content or not markdown_content.strip():
                logger.warning(f"Tệp đính kèm {file_name} không chứa nội dung văn bản khả dụng.")
                return

            # 4. Phân mảnh văn bản sử dụng RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.RAG_CHUNK_SIZE,
                chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_text(markdown_content)
            logger.info(f"Đã chia tài liệu thành {len(chunks)} chunks.")

            # 5. Sinh vector embedding hàng loạt (Batch Embedding)
            try:
                embeddings = await tool_rag.get_embeddings(chunks, provider)
            except Exception as emb_exc:
                logger.error(f"Lỗi sinh batch embedding cho chunks: {emb_exc}. Fallback về sinh đơn lẻ.")
                embeddings = []

            chunk_documents = []
            for idx, chunk in enumerate(chunks):
                emb = embeddings[idx] if idx < len(embeddings) else None
                if not emb:
                    # Fallback sinh đơn lẻ nếu lô thiếu hoặc lỗi
                    emb = await tool_rag.get_embedding(chunk, provider)
                if emb:
                    chunk_documents.append({
                        "chunk_id": f"{file_id}_{idx}",
                        "file_id": file_id,
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "text": chunk,
                        "embedding": emb,
                        "index": idx,
                        "created_at": datetime.now(timezone.utc)
                    })

            # 6. Ghi hàng loạt (Batch Insertion) các chunks vào MongoDB
            if chunk_documents:
                await db["rag_chunks"].insert_many(chunk_documents)
                logger.info(f"Đã ghi hàng loạt {len(chunk_documents)} chunks vào bộ sưu tập rag_chunks.")

            # 7. Lưu trữ tài liệu gốc vào rag_documents
            doc_record = {
                "file_id": file_id,
                "file_name": file_name,
                "file_type": file_type,
                "file_size": file_size,
                "storage_url": storage_url,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "markdown_content": markdown_content,
                "created_at": datetime.now(timezone.utc)
            }
            await db["rag_documents"].insert_one(doc_record)
            logger.info(f"Lưu thành công thông tin tệp {file_name} vào bộ sưu tập rag_documents.")

        except Exception as exc:
            logger.error(f"Lỗi trong quy trình Ingest tệp {file_name} vào RAG: {exc}")
            raise exc

        finally:
            # 8. Xóa file tạm trong khối finally, xử lý ngoại lệ im lặng
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    await asyncio.to_thread(os.remove, temp_file_path)
                    logger.info(f"Đã giải phóng tệp tạm cục bộ: {temp_file_path}")
                except OSError as os_err:
                    logger.warning(f"Không thể xóa tệp tạm tại {temp_file_path}: {os_err}")

    async def search_rag(
        self,
        db: Any,
        query: str,
        conversation_id: str,
        provider: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các đoạn văn bản tương đồng ngữ cảnh trong phạm vi phiên hội thoại.
        """
        if db is None or not query:
            return []

        # 1. Sinh vector embedding cho câu truy vấn
        query_vector = await tool_rag.get_embedding(query, provider)
        if not query_vector:
            logger.warning("Không thể sinh vector cho câu truy vấn. Hủy tìm kiếm RAG.")
            return []

        # 2. Thực hiện Vector Search theo chiến lược được cấu hình
        # PHƯƠNG ÁN A: Sử dụng Native Vector Search trên Cosmos DB vCore
        if settings.VECTOR_SEARCH_PROVIDER == "cosmos":
            try:
                pipeline = [
                    {
                        "$search": {
                            "cosmosSearch": {
                                "vector": query_vector,
                                "path": "embedding",
                                "k": top_k,
                                "filter": {
                                    "conversation_id": conversation_id
                                }
                            }
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "text": 1,
                            "file_id": 1,
                            "score": { "$meta": "searchScore" }
                        }
                    }
                ]
                cursor = db["rag_chunks"].aggregate(pipeline)
                results = await cursor.to_list(length=top_k)
                logger.info(f"RAG search thành công bằng Cosmos DB Native Vector Search (Found: {len(results)}).")
                return results
            except Exception as exc:
                logger.error(f"Lỗi khi thực hiện Cosmos DB Vector Search: {exc}. Fallback về tính toán RAM.")

        # PHƯƠNG ÁN B: Sử dụng Native Vector Search trên MongoDB Atlas
        elif settings.VECTOR_SEARCH_PROVIDER == "atlas":
            try:
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": settings.ATLAS_VECTOR_INDEX_NAME,
                            "path": "embedding",
                            "queryVector": query_vector,
                            "numCandidates": max(100, top_k * settings.NUM_CANDIDATES_FACTOR),
                            "limit": top_k,
                            "filter": {
                                "conversation_id": conversation_id
                            }
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "text": 1,
                            "file_id": 1,
                            "score": { "$meta": "vectorSearchScore" }
                        }
                    }
                ]
                cursor = db["rag_chunks"].aggregate(pipeline)
                results = await cursor.to_list(length=top_k)
                logger.info(f"RAG search thành công bằng MongoDB Atlas Vector Search (Found: {len(results)}).")
                return results
            except Exception as exc:
                logger.error(f"Lỗi khi thực hiện MongoDB Atlas Vector Search: {exc}. Fallback về tính toán RAM.")

        # PHƯƠNG ÁN C: Fallback mặc định - Tính toán Cosine Similarity cục bộ trên RAM bằng numpy
        cursor = db["rag_chunks"].find({"conversation_id": conversation_id})
        chunks = await cursor.to_list(length=1000)

        if not chunks:
            logger.info(f"Không tìm thấy tài liệu đính kèm nào cho phiên hội thoại {conversation_id}.")
            return []

        scored_chunks = []
        for chunk in chunks:
            score = tool_rag.cosine_similarity(query_vector, chunk.get("embedding", []))
            scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_results = []
        for chunk, score in scored_chunks[:top_k]:
            top_results.append({
                "text": chunk["text"],
                "file_id": chunk["file_id"],
                "score": score
            })

        logger.info(f"Tìm kiếm RAG hoàn tất bằng numpy fallback. Lấy được {len(top_results)} kết quả tương đồng.")
        return top_results

document_rag = DocumentRAGService()
