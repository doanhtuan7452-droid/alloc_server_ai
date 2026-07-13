from fastapi import APIRouter, HTTPException, Depends, Request
from app.schemas.project_risk import ProjectData
from app.services.project_risk import project_risk_service
from app.core.auth import get_api_key
from app.core.rate_limit import limiter

router = APIRouter(dependencies=[Depends(get_api_key)])

@router.post("/predict-risk")
@limiter.limit("30/minute")
def predict_project_risk(request: Request, data: ProjectData):
    try:
        return project_risk_service.predict(data)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        raise exc
