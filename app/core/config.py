import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load env variables at the very beginning of configuration initialization
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

class Settings:
    # Root directory of the project
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # Employee Recommendation model paths
    EMPLOYEE_MODEL_DIR: Path = BASE_DIR / "model_ai" / "employee_recommendation"
    EMPLOYEE_MODEL_PATH: Path = EMPLOYEE_MODEL_DIR / "hr_allocation_ai_model.pkl"
    EMPLOYEE_SCALER_PATH: Path = EMPLOYEE_MODEL_DIR / "hr_scaler.pkl"
    EMPLOYEE_LABEL_ENCODER_PATH: Path = EMPLOYEE_MODEL_DIR / "hr_label_encoder.pkl"
    
    # Project Risk model paths
    PROJECT_RISK_MODEL_DIR: Path = BASE_DIR / "model_ai" / "project_risk"
    PROJECT_RISK_MODEL_PATH: Path = PROJECT_RISK_MODEL_DIR / "project_risk_model.pkl"
    PROJECT_RISK_SCALER_PATH: Path = PROJECT_RISK_MODEL_DIR / "project_scaler.pkl"
    
    # Fit Regressor model paths
    FIT_REGRESSOR_MODEL_DIR: Path = BASE_DIR / "model_ai" / "fit_regressor"
    FIT_REGRESSOR_MODEL_PATH: Path = FIT_REGRESSOR_MODEL_DIR / "hr_fit_regressor_model.pkl"
    FIT_REGRESSOR_SCALER_PATH: Path = FIT_REGRESSOR_MODEL_DIR / "hr_fit_scaler.pkl"
    
    # Feature columns used for fit percentage prediction
    FIT_REGRESSOR_FEATURE_COLUMNS: List[str] = [
        "experience_years",
        "education_level",
        "skill_level",
        "technical_skill_score",
        "communication_score",
        "leadership_score",
        "problem_solving_score",
        "task_complexity",
        "required_skill_level",
        "deadline_days",
        "workload_hours",
        "task_priority",
        "team_size",
        "attendance_rate",
        "performance_rating",
        "conflict_rate",
        "hours_per_day",
        "skill_gap",
        "avg_soft_skill",
    ]

    # Feature columns used for employee suggestion mapping
    ALLOCATION_FEATURE_COLUMNS: List[str] = [
        "experience_years",
        "education_level",
        "skill_level",
        "technical_skill_score",
        "communication_score",
        "leadership_score",
        "problem_solving_score",
        "task_complexity",
        "required_skill_level",
        "deadline_days",
        "workload_hours",
        "task_priority",
        "team_size",
        "attendance_rate",
        "performance_rating",
        "conflict_rate",
        "skill_gap",
        "hours_per_day",
        "avg_soft_skill",
    ]
    
    # Feature columns used for project risk prediction mapping
    PROJECT_RISK_FEATURE_COLUMNS: List[str] = [
        "Project_Duration_Days",
        "Expected_Budget",
        "Team_Size",
        "Avg_Team_Skill_Level",
        "Complexity_Score",
        "Budget_Utilization",
        "Methodology_Used_Hybrid",
        "Methodology_Used_Kanban",
        "Methodology_Used_Scrum",
        "Methodology_Used_Waterfall",
    ]

    # LLM & Agent Configurations
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # API Keys & Endpoints
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # MongoDB Configuration
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "ai_allocation_db")
    MONGODB_TLS: bool = os.getenv("MONGODB_TLS", "false").lower() in ("true", "1", "yes")
    MONGODB_TLS_CA_FILE: Optional[str] = os.getenv("MONGODB_TLS_CA_FILE", None)
    MONGODB_RETRY_WRITES: bool = os.getenv("MONGODB_RETRY_WRITES", "true").lower() in ("true", "1", "yes")
    
    # Vector Search Configurations
    # Hỗ trợ: "numpy" (tính trên RAM - mặc định), "cosmos" (Cosmos DB vCore), "atlas" (MongoDB Atlas)
    VECTOR_SEARCH_PROVIDER: str = os.getenv("VECTOR_SEARCH_PROVIDER", "numpy").lower()
    ATLAS_VECTOR_INDEX_NAME: str = os.getenv("ATLAS_VECTOR_INDEX_NAME", "default")
    RAG_EMBEDDING_DIMENSIONS: int = int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "1536"))
    NUM_CANDIDATES_FACTOR: int = int(os.getenv("NUM_CANDIDATES_FACTOR", "20"))

    # Security Configuration
    ALLOW_ANONYMOUS_CHAT: bool = os.getenv("ALLOW_ANONYMOUS_CHAT", "true").lower() in ("true", "1", "yes")
    PYTHON_API_KEY: str = os.getenv("PYTHON_API_KEY", "testkey123")
    INTERNAL_API_SECRET: str = os.getenv("INTERNAL_API_SECRET", "internal_secret_key_123")
    ALLOWED_INTERNAL_IPS: List[str] = [
        ip.strip() for ip in os.getenv("ALLOWED_INTERNAL_IPS", "127.0.0.1,::1,localhost").split(",") if ip.strip()
    ]

    # Search & Tool RAG Configurations
    SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "duckduckgo")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    GOOGLE_SEARCH_API_KEY: str = os.getenv("GOOGLE_SEARCH_API_KEY", "")
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")
    SEARCH_MAX_RESULTS: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
    TOOL_RAG_TOP_K: int = int(os.getenv("TOOL_RAG_TOP_K", "4"))
    TOOL_RAG_THRESHOLD: float = float(os.getenv("TOOL_RAG_THRESHOLD", "0.35"))

    # C# Server API configurations
    CSHARP_API_BASE_URL: str = os.getenv("CSHARP_API_BASE_URL", "http://localhost:5000")
    CSHARP_FILE_DOWNLOAD_PATH: str = os.getenv("CSHARP_FILE_DOWNLOAD_PATH", "/api/v1/files/download/{file_id}")

    # Document RAG Configurations
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    RAG_MAX_RESULTS: int = int(os.getenv("RAG_MAX_RESULTS", "5"))
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    RAG_ALLOWED_EXTENSIONS: List[str] = [
        ext.strip() for ext in os.getenv(
            "RAG_ALLOWED_EXTENSIONS", ".pdf,.docx,.xlsx,.pptx,.txt,.csv,.html,.md,.json"
        ).lower().split(",") if ext.strip()
    ]

    # Environment
    ENV: str = os.getenv("ENV", "development").lower()

    # Redis backend configuration for multi-worker SlowAPI rate limiting
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL", None)

    # Allowed origins for CORS (comma-separated)
    ALLOWED_ORIGINS: List[str] = [
        origin.strip() for origin in os.getenv(
            "ALLOWED_ORIGINS", 
            "https://csharp-backend-production.azurewebsites.net,http://localhost:5000"
        ).split(",") if origin.strip()
    ]

    # Allowed domains for file downloads (SSRF protection, comma-separated)
    ALLOWED_STORAGE_DOMAINS: List[str] = [
        domain.strip() for domain in os.getenv(
            "ALLOWED_STORAGE_DOMAINS",
            "mycompany.blob.core.windows.net,localhost,127.0.0.1"
        ).split(",") if domain.strip()
    ]

    # CA Bundle path or boolean for C# storage certificate verification
    CSHARP_STORAGE_CA_BUNDLE: str = os.getenv("CSHARP_STORAGE_CA_BUNDLE", "true")

    # Trusted proxy IPs for Uvicorn (comma-separated, default loopback)
    FORWARDED_ALLOW_IPS: str = os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1,::1")

    # Whitelisted header names for dynamic tools metadata
    ALLOWED_DYNAMIC_HEADERS: List[str] = [
        h.strip().lower() for h in os.getenv(
            "ALLOWED_DYNAMIC_HEADERS",
            "authorization,content-type,accept,x-workspace-id,x-user-id"
        ).split(",") if h.strip()
    ]

    def __init__(self):
        # Tự động chuẩn hóa MongoDB URL khi khởi tạo cấu hình
        self.MONGODB_URL = self._escape_mongodb_url(self.MONGODB_URL)

    @staticmethod
    def _escape_mongodb_url(url: str) -> str:
        import urllib.parse
        if not (url.startswith("mongodb://") or url.startswith("mongodb+srv://")):
            return url
        
        prefix = "mongodb+srv://" if url.startswith("mongodb+srv://") else "mongodb://"
        rest = url[len(prefix):]
        
        # Tách phần options (sau dấu '?') khỏi phần còn lại của URI
        if '?' in rest:
            main_part, options = rest.split('?', 1)
            options_str = '?' + options
        else:
            main_part = rest
            options_str = ''
            
        # Tìm dấu ngăn cách credentials (dấu '@' cuối cùng trong main_part)
        if '@' not in main_part:
            return url
            
        creds, host_and_path = main_part.rsplit('@', 1)
        
        if ':' in creds:
            username, password = creds.split(':', 1)
        else:
            username = creds
            password = ""
            
        # Giải mã trước để tránh double-encoding, sau đó mã hóa chuẩn RFC 3986
        escaped_username = urllib.parse.quote_plus(urllib.parse.unquote(username))
        escaped_password = urllib.parse.quote_plus(urllib.parse.unquote(password))
        
        if escaped_password:
            escaped_creds = f"{escaped_username}:{escaped_password}"
        else:
            escaped_creds = escaped_username
            
        return f"{prefix}{escaped_creds}@{host_and_path}{options_str}"


settings = Settings()

# Validate that INTERNAL_API_SECRET is configured at startup to fail-fast on misconfigured servers.
if not settings.INTERNAL_API_SECRET or not settings.INTERNAL_API_SECRET.strip():
    raise ValueError("CRITICAL CONFIGURATION ERROR: INTERNAL_API_SECRET environment variable is missing or empty. Server startup aborted.")


