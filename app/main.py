import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.types import Message
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.router import api_router
from app.services.employee import employee_service
from app.services.project_risk import project_risk_service
from app.services.fit_regressor import fit_regressor_service
from app.core.database import init_db, close_db
from app.core.config import settings
from app.core.rate_limit import limiter
from app.core.exceptions import (
    global_unhandled_exception_handler,
    custom_validation_exception_handler
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model binaries on application startup
    employee_service.load_models()
    project_risk_service.load_models()
    fit_regressor_service.load_models()
    # Initialize MongoDB connection and indexes
    await init_db(app)
    yield
    # Close MongoDB client
    await close_db()

# Hide Swagger UI and OpenAPI docs on production
app = FastAPI(
    title="Employee Suggestion AI API",
    description="API for C# server integration with employee allocation and project risk models.",
    version="1.0",
    lifespan=lifespan,
    docs_url=None if settings.ENV == "production" else "/docs",
    redoc_url=None if settings.ENV == "production" else "/redoc",
    openapi_url=None if settings.ENV == "production" else "/openapi.json"
)

# Set up SlowAPI Limiter state
app.state.limiter = limiter

# Register Global Exception Handlers
app.add_exception_handler(Exception, global_unhandled_exception_handler)
app.add_exception_handler(RequestValidationError, custom_validation_exception_handler)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware (low security priority headless API server configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Body caching middleware for extracting Rate Limit keys from request body
@app.middleware("http")
async def cache_body_middleware(request: Request, call_next):
    # Only cache body for JSON requests to POST/PUT endpoints to avoid overhead or blocking large stream loads
    content_type = request.headers.get("content-type", "")
    if request.method in ("POST", "PUT") and "application/json" in content_type:
        try:
            body = await request.body()
            request.state.body_json = json.loads(body) if body else {}
            
            # Re-inject the body stream so downstream FastAPI parameters extraction still works
            async def receive() -> Message:
                return {"type": "http.request", "body": body}
            request._receive = receive
        except Exception:
            request.state.body_json = {}
    else:
        request.state.body_json = {}
        
    return await call_next(request)

@app.get("/")
def read_root():
    return {
        "status": "Server đang hoạt động bình thường",
        "docs_url": "Hãy truy cập http://127.0.0.1:8000/docs để test API" if settings.ENV != "production" else "Tài liệu API đã được vô hiệu hóa trên môi trường này.",
    }

# Include backward-compatible routes
app.include_router(api_router)
