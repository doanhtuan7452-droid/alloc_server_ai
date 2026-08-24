from fastapi import APIRouter, HTTPException, Depends, Body, Request
from typing import Union
from app.schemas.assessment import (
    PersonnelAssessmentRequest,
    BulkAssessmentRequest,
    AllocationAssessmentResponse,
    BulkAllocationAssessmentResponse,
    ProjectRiskAssessmentRequest,
    ProjectRiskAssessmentResponse
)
from app.services.allocation_assessment import allocation_assessment_service
from app.services.project_risk_assessment import project_risk_assessment_service
from app.core.auth import get_api_key
from app.core.rate_limit import limiter

router = APIRouter(dependencies=[Depends(get_api_key)])

@router.post(
    "/allocation/assess",
    response_model=Union[AllocationAssessmentResponse, BulkAllocationAssessmentResponse],
    summary="Đánh giá sự phù hợp và rủi ro của nhân sự được phân bổ"
)
@limiter.limit("20/minute")
async def assess_personnel_allocation(
    request: Request,
    payload: Union[PersonnelAssessmentRequest, BulkAssessmentRequest] = Body(...)
):
    try:
        db = getattr(request.app.state, "db", None) if hasattr(request, "app") and hasattr(request.app, "state") else None
        return await allocation_assessment_service.assess(payload, db=db)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        raise exc

@router.post(
    "/project-risk/assess",
    response_model=ProjectRiskAssessmentResponse,
    summary="Đánh giá mức độ rủi ro của dự án và giải thích"
)
@limiter.limit("20/minute")
async def assess_project_risk(request: Request, payload: ProjectRiskAssessmentRequest):
    try:
        return await project_risk_assessment_service.assess(payload)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        raise exc
