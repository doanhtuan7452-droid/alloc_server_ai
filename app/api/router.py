from fastapi import APIRouter
from app.api.endpoints.allocation import router as allocation_router
from app.api.endpoints.project_risk import router as risk_router

api_router = APIRouter()

# Group endpoints under /api/v1 prefix with NO tags configuration
api_router.include_router(allocation_router, prefix="/api/v1")
api_router.include_router(risk_router, prefix="/api/v1")
