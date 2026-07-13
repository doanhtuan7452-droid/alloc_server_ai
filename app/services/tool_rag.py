import re
import asyncio
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings

logger = logging.getLogger("app.services.tool_rag")

class ToolRAGService:
    def __init__(self):
        # Cache bộ đệm cho embeddings của local/static tools
        # Cấu trúc: { provider: { tool_name: list_of_floats } }
        self._static_embeddings_cache = {}

    def clean_and_truncate_context(self, text: str) -> str:
        """
        Làm sạch markdown (code blocks, tables, links) và cắt ngắn phản hồi trước đó
        để lấy tối đa 2 câu đầu tiên hoặc 50 từ đầu tiên nhằm tránh làm loãng vector.
        """
        if not text:
            return ""
        try:
            # Xóa các khối code (codeblocks)
            text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
            # Xóa các dòng bảng biểu markdown
            text = re.sub(r'\|.*?\|', '', text)
            # Xóa liên kết markdown [text](url)
            text = re.sub(r'\[.*?\]\(.*?\)', '', text)
            # Xóa các ký tự markdown thông thường
            text = re.sub(r'[*#_`~]', '', text)
            # Chuẩn hóa khoảng trắng
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                return ""

            # Tách thành các câu dựa trên dấu chấm/hỏi/chấm than kèm khoảng trắng
            sentences = re.split(r'(?<=[.!?])\s+', text)
            selected_sentences = sentences[:2]
            joined = " ".join(selected_sentences)
            
            # Cắt tối đa 50 từ
            words = joined.split()
            if len(words) > 50:
                joined = " ".join(words[:50])
                
            # Cắt tối đa 200 ký tự để chắc chắn không bị quá dài
            if len(joined) > 200:
                joined = joined[:200]
                
            return joined
        except Exception as e:
            logger.error(f"Lỗi khi làm sạch context: {e}")
            return text[:100]

    def enrich_query(self, query: str, last_response: Optional[str]) -> str:
        """
        Gộp câu hỏi hiện tại với ngữ cảnh làm sạch từ câu trả lời trước đó.
        """
        cleaned_context = self.clean_and_truncate_context(last_response or "")
        if cleaned_context:
            enriched = f"[Context: {cleaned_context}] {query}"
            logger.info(f"Query đã làm giàu: {enriched}")
            return enriched
        return query

    def _get_embedding_client(self, provider: str) -> Any:
        """Khởi tạo embedding client thích hợp dựa trên provider."""
        prov = provider.lower()
        if prov == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY không được cấu hình.")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=settings.OPENAI_API_KEY
            )
        elif prov == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY không được cấu hình.")
            return GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=settings.GEMINI_API_KEY
            )
        elif prov == "ollama":
            return OllamaEmbeddings(
                model=settings.OLLAMA_EMBEDDING_MODEL,
                base_url=settings.OLLAMA_BASE_URL
            )
        else:
            raise ValueError(f"Không hỗ trợ sinh embedding cho provider: {provider}")

    async def get_embedding(self, text: str, provider: str) -> List[float]:
        """Sinh vector embedding cho một chuỗi văn bản."""
        try:
            client = self._get_embedding_client(provider)
            # invoke/embed_query
            embedding = await client.aembed_query(text)
            return [float(x) for x in embedding]
        except Exception as e:
            logger.error(f"Lỗi khi sinh embedding cho text: {e}")
            # Trả về vector rỗng nếu lỗi
            return []

    async def get_embeddings(self, texts: List[str], provider: str) -> List[List[float]]:
        """Sinh vector embedding cho danh sách các chuỗi văn bản bằng batch embedding."""
        if not texts:
            return []
        
        # Batch Chunk Slicing: Chia danh sách thành các lô nhỏ tối đa 32 chunks
        batch_size = 32
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        try:
            client = self._get_embedding_client(provider)
            all_embeddings = []
            
            # Gửi song song các lô bằng asyncio.gather để tối ưu hóa hiệu năng
            async def embed_batch(batch):
                return await client.aembed_documents(batch)
                
            tasks = [embed_batch(b) for b in batches]
            results = await asyncio.gather(*tasks)
            
            for res in results:
                for emb in res:
                    all_embeddings.append([float(x) for x in emb])
                    
            return all_embeddings
        except Exception as e:
            logger.error(f"Lỗi khi sinh batch embedding: {e}")
            return []

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Tính toán khoảng cách Cosine giữa hai vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        try:
            arr1 = np.array(v1, dtype=float)
            arr2 = np.array(v2, dtype=float)
            dot_product = np.dot(arr1, arr2)
            norm_v1 = np.linalg.norm(arr1)
            norm_v2 = np.linalg.norm(arr2)
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            return float(dot_product / (norm_v1 * norm_v2))
        except Exception as e:
            logger.error(f"Lỗi tính toán cosine similarity: {e}")
            return 0.0

    async def get_or_create_static_tool_embedding(self, tool_name: str, description: str, provider: str) -> List[float]:
        """Lấy embedding của local tool từ cache hoặc sinh mới nếu chưa có."""
        prov = provider.lower()
        if prov not in self._static_embeddings_cache:
            self._static_embeddings_cache[prov] = {}
        
        if tool_name not in self._static_embeddings_cache[prov]:
            logger.info(f"Sinh embedding mới cho local tool: {tool_name} (provider: {prov})")
            emb = await self.get_embedding(description, prov)
            if emb:
                self._static_embeddings_cache[prov][tool_name] = emb
            else:
                return []
        
        return self._static_embeddings_cache[prov].get(tool_name, [])

    async def get_or_create_static_tool_embedding_db(
        self,
        db: Any,
        tool_name: str,
        description: str,
        provider: str
    ) -> List[float]:
        """Lấy embedding của local/static tool từ DB hoặc sinh mới và lưu vào DB nếu chưa có."""
        from datetime import datetime, timezone
        prov = provider.lower()
        if db is None:
            # Fallback to RAM cache if db is not provided
            return await self.get_or_create_static_tool_embedding(tool_name, description, provider)
            
        # 1. Check RAM cache first
        if prov not in self._static_embeddings_cache:
            self._static_embeddings_cache[prov] = {}
        if tool_name in self._static_embeddings_cache[prov]:
            return self._static_embeddings_cache[prov][tool_name]
            
        # 2. Check MongoDB L2 Cache
        try:
            doc = await db["static_tool_embeddings"].find_one({"tool_name": tool_name, "provider": prov})
            if doc and "embedding" in doc and doc["embedding"]:
                emb = [float(x) for x in doc["embedding"]]
                self._static_embeddings_cache[prov][tool_name] = emb
                logger.info(f"Lấy thành công embedding của static tool {tool_name} từ MongoDB L2 cache.")
                return emb
        except Exception as e:
            logger.warning(f"Lỗi khi truy vấn static tool embedding từ DB: {e}")
            
        # 3. Generate via API
        emb = await self.get_embedding(description, prov)
        if emb:
            self._static_embeddings_cache[prov][tool_name] = emb
            try:
                await db["static_tool_embeddings"].update_one(
                    {"tool_name": tool_name, "provider": prov},
                    {"$set": {
                        "tool_name": tool_name,
                        "provider": prov,
                        "description": description,
                        "embedding": emb,
                        "updated_at": datetime.now(timezone.utc)
                    }},
                    upsert=True
                )
                logger.info(f"Đã sinh mới và lưu embedding của static tool {tool_name} vào MongoDB.")
            except Exception as e:
                logger.error(f"Lỗi khi lưu static tool embedding vào DB: {e}")
            return emb
        return []

tool_rag = ToolRAGService()
