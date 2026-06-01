from fastapi import APIRouter, HTTPException
from app.schemas.project_risk import ProjectData
from app.services.project_risk import project_risk_service

router = APIRouter()

@router.post("/predict-risk")
def predict_project_risk(data: ProjectData):
    try:
        return project_risk_service.predict(data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
