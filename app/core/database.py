from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import logging

logger = logging.getLogger("app.database")

class Database:
    client: AsyncIOMotorClient = None

db_instance = Database()

async def init_db(app):
    """Khởi tạo kết nối MongoDB và tạo chỉ mục"""
    try:
        client_kwargs = {
            "serverSelectionTimeoutMS": 2000,
            "maxPoolSize": 50,
            "minPoolSize": 10,
            "maxIdleTimeMS": 10000,
            "waitQueueTimeoutMS": 5000
        }
        
        # Cấu hình TLS/SSL động cho Azure Cosmos DB hoặc MongoDB Atlas
        if settings.MONGODB_TLS:
            client_kwargs["tls"] = True
            if settings.MONGODB_TLS_CA_FILE:
                client_kwargs["tlsCAFile"] = settings.MONGODB_TLS_CA_FILE
            else:
                try:
                    import certifi
                    client_kwargs["tlsCAFile"] = certifi.where()
                except ImportError:
                    logger.warning("Thư viện certifi chưa được cài đặt. Không thể cấu hình tlsCAFile.")
        
        # Cấu hình retryWrites động (Cosmos vCore yêu cầu False, Atlas yêu cầu True)
        client_kwargs["retryWrites"] = settings.MONGODB_RETRY_WRITES

        db_instance.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            **client_kwargs
        )
        app.state.db = db_instance.client[settings.MONGODB_DB_NAME]
        
        # Kiểm tra kết nối thực tế
        await db_instance.client.admin.command('ping')
        
        # Tạo index tự động bảo vệ hiệu năng
        await app.state.db["conversations"].create_index("conversation_id", unique=True)
        await app.state.db["conversations"].create_index("user_id")
        await app.state.db["conversations"].create_index([("user_id", 1), ("updated_at", -1)])
        await app.state.db["messages"].create_index([("conversation_id", 1), ("timestamp", -1)])
        
        # Document RAG Indexes
        await app.state.db["rag_documents"].create_index("file_id", unique=True)
        await app.state.db["rag_chunks"].create_index([("conversation_id", 1), ("file_id", 1)])
        
        # Khởi tạo Native Vector Index trên Cosmos DB vCore nếu cấu hình yêu cầu
        if settings.VECTOR_SEARCH_PROVIDER == "cosmos":
            try:
                existing_indexes = await app.state.db["rag_chunks"].index_information()
                if "rag_chunks_vector_index" not in existing_indexes:
                    await app.state.db.command({
                        "createIndexes": "rag_chunks",
                        "indexes": [
                            {
                                "name": "rag_chunks_vector_index",
                                "key": {
                                    "embedding": "cosmosSearch"
                                },
                                "cosmosSearchOptions": {
                                    "kind": "vector-hnsw",
                                    "dimensions": settings.RAG_EMBEDDING_DIMENSIONS,
                                    "similarity": "COS",
                                    "m": 16,
                                    "efConstruction": 64
                                }
                            }
                        ]
                    })
                    logger.info("Native Vector Index created on Cosmos DB vCore.")
            except Exception as index_exc:
                logger.warning(f"Không thể khởi tạo Cosmos DB Vector Index: {index_exc}. Fallback về chạy numpy.")
        
        # TTL Indexes to automatically delete documents/chunks after 30 days (2,592,000 seconds)
        await app.state.db["rag_documents"].create_index("created_at", expireAfterSeconds=2592000)
        await app.state.db["rag_chunks"].create_index("created_at", expireAfterSeconds=2592000)
        
        logger.info("MongoDB Connection Initialized & Indexes Created.")
    except Exception as exc:
        logger.error(f"Failed to connect to MongoDB: {exc}. Server will run without database features.")
        app.state.db = None


async def close_db():
    """Đóng kết nối MongoDB"""
    if db_instance.client:
        db_instance.client.close()
        logger.info("MongoDB Connection Closed.")
