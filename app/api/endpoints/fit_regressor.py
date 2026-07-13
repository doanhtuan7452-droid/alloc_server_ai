from fastapi import APIRouter, HTTPException, Depends, Request
from app.schemas.fit_regressor import PredictionRequest, PredictionResponse
from app.services.fit_regressor import fit_regressor_service
from app.core.auth import get_api_key
from app.core.rate_limit import limiter

router = APIRouter(dependencies=[Depends(get_api_key)])

@router.post("/allocation/predict-fit", response_model=PredictionResponse)
@limiter.limit("30/minute")
def predict_fit_percentage(request: Request, predict_req: PredictionRequest):
    try:
        return fit_regressor_service.predict(predict_req)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        raise exc
