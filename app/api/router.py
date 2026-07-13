from fastapi import APIRouter
from app.api.endpoints.allocation import router as allocation_router
from app.api.endpoints.project_risk import router as risk_router
from app.api.endpoints.chat import router as chat_router
from app.api.endpoints.mongo_chat import router as mongo_chat_router
from app.api.endpoints.assessment import router as assessment_router
from app.api.endpoints.tools import router as tools_router
from app.api.endpoints.fit_regressor import router as fit_regressor_router
from app.api.endpoints.internal_chat import router as internal_chat_router

api_router = APIRouter()

# Group endpoints under /api/v1 prefix with NO tags configuration
api_router.include_router(allocation_router, prefix="/api/v1")
api_router.include_router(risk_router, prefix="/api/v1")
api_router.include_router(chat_router, prefix="/api/v1")
api_router.include_router(mongo_chat_router, prefix="/api/v1")
api_router.include_router(assessment_router, prefix="/api/v1")
api_router.include_router(tools_router, prefix="/api/v1")
api_router.include_router(fit_regressor_router, prefix="/api/v1")

# Group internal endpoints under /internal/v1 prefix
api_router.include_router(internal_chat_router, prefix="/internal/v1")


