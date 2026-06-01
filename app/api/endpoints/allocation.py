from fastapi import APIRouter, HTTPException
from app.schemas.allocation import AllocationRequest
from app.services.employee import employee_service

router = APIRouter()

@router.post("/suggest-allocation")
def suggest_allocation(request: AllocationRequest):
    try:
        return employee_service.predict(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
