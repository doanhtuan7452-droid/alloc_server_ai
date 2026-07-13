from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Import Config Settings
from app.core.config import settings

# Re-export path variables and configurations
BASE_DIR = settings.BASE_DIR

EMPLOYEE_MODEL_DIR = settings.EMPLOYEE_MODEL_DIR
EMPLOYEE_MODEL_PATH = settings.EMPLOYEE_MODEL_PATH
EMPLOYEE_SCALER_PATH = settings.EMPLOYEE_SCALER_PATH
EMPLOYEE_LABEL_ENCODER_PATH = settings.EMPLOYEE_LABEL_ENCODER_PATH

PROJECT_RISK_MODEL_DIR = settings.PROJECT_RISK_MODEL_DIR
PROJECT_RISK_MODEL_PATH = settings.PROJECT_RISK_MODEL_PATH
PROJECT_RISK_SCALER_PATH = settings.PROJECT_RISK_SCALER_PATH

FIT_REGRESSOR_MODEL_DIR = settings.FIT_REGRESSOR_MODEL_DIR
FIT_REGRESSOR_MODEL_PATH = settings.FIT_REGRESSOR_MODEL_PATH
FIT_REGRESSOR_SCALER_PATH = settings.FIT_REGRESSOR_SCALER_PATH

ALLOCATION_FEATURE_COLUMNS = settings.ALLOCATION_FEATURE_COLUMNS
PROJECT_RISK_FEATURE_COLUMNS = settings.PROJECT_RISK_FEATURE_COLUMNS
FIT_REGRESSOR_FEATURE_COLUMNS = settings.FIT_REGRESSOR_FEATURE_COLUMNS

# 2. Utility Function Re-exports
def model_to_dict(data: BaseModel) -> dict:
    if hasattr(data, "model_dump"):
        return data.model_dump()
    return data.dict()

def load_joblib(path: Path, name: str):
    import joblib
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"[!] Could not load {name} from {path}: {exc}")
        return None

# 3. Import Pydantic Schemas
from app.schemas.allocation import AllocationRequest
from app.schemas.project_risk import ProjectData

# 4. Import Services (so we can pre-populate them on import-time loading)
from app.services.employee import employee_service
from app.services.project_risk import project_risk_service
from app.services.fit_regressor import fit_regressor_service

# 5. Load model binaries immediately at module import time to maintain import-time properties
import joblib

try:
    employee_model = joblib.load(EMPLOYEE_MODEL_PATH)
    employee_scaler = joblib.load(EMPLOYEE_SCALER_PATH)
    employee_label_encoder = joblib.load(EMPLOYEE_LABEL_ENCODER_PATH)
    
    # Pre-populate service to prevent duplicate loading on lifespan startup
    employee_service.model = employee_model
    employee_service.scaler = employee_scaler
    employee_service.label_encoder = employee_label_encoder
except Exception as exc:
    print(f"[!] Could not load employee recommendation models at module import: {exc}")
    employee_model = None
    employee_scaler = None
    employee_label_encoder = None

try:
    project_risk_model = joblib.load(PROJECT_RISK_MODEL_PATH)
    project_risk_scaler = joblib.load(PROJECT_RISK_SCALER_PATH)
    
    # Pre-populate service to prevent duplicate loading on lifespan startup
    project_risk_service.model = project_risk_model
    project_risk_service.scaler = project_risk_scaler
except Exception as exc:
    print(f"[!] Could not load project risk models at module import: {exc}")
    project_risk_model = None
    project_risk_scaler = None

try:
    fit_regressor_model = joblib.load(FIT_REGRESSOR_MODEL_PATH)
    fit_regressor_scaler = joblib.load(FIT_REGRESSOR_SCALER_PATH)
    
    # Pre-populate service to prevent duplicate loading on lifespan startup
    fit_regressor_service.model = fit_regressor_model
    fit_regressor_service.scaler = fit_regressor_scaler
except Exception as exc:
    print(f"[!] Could not load fit regressor models at module import: {exc}")
    fit_regressor_model = None
    fit_regressor_scaler = None

# 6. Re-export Assertions
def ensure_employee_models_loaded() -> None:
    if (
        employee_service.model is None
        or employee_service.scaler is None
        or employee_service.label_encoder is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Employee recommendation model is not loaded.",
        )

def ensure_project_risk_models_loaded() -> None:
    if project_risk_service.model is None or project_risk_service.scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Project risk model is not loaded.",
        )

def ensure_fit_regressor_models_loaded() -> None:
    if fit_regressor_service.model is None or fit_regressor_service.scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Fit regressor model is not loaded.",
        )

# 7. Import real app instance and expose it
from app.main import app

# 8. Re-export route functions
from app.api.endpoints.allocation import suggest_allocation
from app.api.endpoints.project_risk import predict_project_risk
from app.api.endpoints.assessment import assess_personnel_allocation, assess_project_risk
from app.api.endpoints.fit_regressor import predict_fit_percentage
