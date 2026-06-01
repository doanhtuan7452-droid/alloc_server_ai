import joblib
import pandas as pd
from fastapi import HTTPException
from app.core.config import settings
from app.schemas.project_risk import ProjectData

class ProjectRiskService:
    def __init__(self):
        self.model = None
        self.scaler = None

    def load_models(self):
        """Load the ML pipeline components into memory. Executed at startup."""
        if self.model is not None and self.scaler is not None:
            return
        try:
            self.model = joblib.load(settings.PROJECT_RISK_MODEL_PATH)
            self.scaler = joblib.load(settings.PROJECT_RISK_SCALER_PATH)
            print("[+] Project risk models loaded successfully.")
        except Exception as exc:
            print(f"[!] Could not load project risk models: {exc}")

    def ensure_loaded(self):
        if self.model is None or self.scaler is None:
            raise HTTPException(
                status_code=503,
                detail="Project risk model is not loaded.",
            )

    def predict(self, data: ProjectData) -> dict:
        self.ensure_loaded()

        input_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[settings.PROJECT_RISK_FEATURE_COLUMNS]

        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)[0]
        risk_status = "High/Critical Risk" if prediction == 1 else "Low/Medium Risk"

        return {
            "prediction_code": int(prediction),
            "risk_status": risk_status,
            "message": "Dự đoán thành công",
        }

project_risk_service = ProjectRiskService()
