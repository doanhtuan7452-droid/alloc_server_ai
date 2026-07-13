from fastapi import APIRouter, HTTPException, Depends, Request
from app.schemas.allocation import AllocationRequest
from app.services.employee import employee_service
from app.core.auth import get_api_key
from app.core.rate_limit import limiter

router = APIRouter(dependencies=[Depends(get_api_key)])

@router.post("/suggest-allocation")
@limiter.limit("30/minute")
def suggest_allocation(request: Request, alloc_req: AllocationRequest):
    try:
        return employee_service.predict(alloc_req)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        raise exc
