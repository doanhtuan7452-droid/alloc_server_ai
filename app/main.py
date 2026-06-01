from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.router import api_router
from app.services.employee import employee_service
from app.services.project_risk import project_risk_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model binaries on application startup
    employee_service.load_models()
    project_risk_service.load_models()
    yield

app = FastAPI(
    title="Employee Suggestion AI API",
    description="API for C# server integration with employee allocation and project risk models.",
    version="1.0",
    lifespan=lifespan,
)

@app.get("/")
def read_root():
    return {
        "status": "Server đang hoạt động bình thường",
        "docs_url": "Hãy truy cập http://127.0.0.1:8000/docs để test API",
    }

# Include perfect backward-compatible routes
app.include_router(api_router)
