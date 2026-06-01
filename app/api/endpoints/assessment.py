from fastapi import APIRouter, HTTPException
from app.schemas.assessment import (
    PersonnelAssessmentRequest,
    AllocationAssessmentResponse,
    ProjectRiskAssessmentRequest,
    ProjectRiskAssessmentResponse
)
from app.services.allocation_assessment import allocation_assessment_service
from app.services.project_risk_assessment import project_risk_assessment_service

router = APIRouter()

@router.post(
    "/allocation/assess",
    response_model=AllocationAssessmentResponse,
    summary="Đánh giá sự phù hợp và rủi ro của nhân sự được phân bổ"
)
async def assess_personnel_allocation(request: PersonnelAssessmentRequest):
    try:
        return await allocation_assessment_service.assess(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.post(
    "/project-risk/assess",
    response_model=ProjectRiskAssessmentResponse,
    summary="Đánh giá mức độ rủi ro của dự án và giải thích"
)
async def assess_project_risk(request: ProjectRiskAssessmentRequest):
    try:
        return await project_risk_assessment_service.assess(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
