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

    def prepare_features(self, data: ProjectData) -> pd.DataFrame:
        """Convert input data into a DataFrame and filter feature columns."""
        input_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Normalize Budget_Utilization if sent as percentage (e.g. > 2.0)
        if "Budget_Utilization" in input_df.columns:
            val = input_df["Budget_Utilization"].iloc[0]
            if val > 2.0:
                input_df["Budget_Utilization"] = float(val / 100.0)
                
        return input_df[settings.PROJECT_RISK_FEATURE_COLUMNS]

    def predict_with_probabilities(self, data: ProjectData) -> dict:
        """Runs inference and returns detailed project risk predictions including class probabilities."""
        self.ensure_loaded()
        input_df = self.prepare_features(data)

        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)[0]
        proba = self.model.predict_proba(input_scaled)[0]
        
        # Mapping classes using self.model.classes_ dynamically
        model_classes = self.model.classes_
        risk_labels_map = {0: "Low/Medium Risk", 1: "High/Critical Risk"}
        class_probabilities = {
            risk_labels_map[int(c)]: float(prob)
            for c, prob in zip(model_classes, proba)
        }
        
        # Confidence score (0.0 to 100.0 scale)
        if prediction == 1:
            confidence_score = class_probabilities.get("High/Critical Risk", 0.0) * 100.0
        else:
            confidence_score = class_probabilities.get("Low/Medium Risk", 0.0) * 100.0
            
        risk_status = "High/Critical Risk" if prediction == 1 else "Low/Medium Risk"

        return {
            "prediction_label": risk_status,
            "prediction_code": int(prediction),
            "class_probabilities": class_probabilities,
            "confidence_score": confidence_score
        }

    def predict(self, data: ProjectData) -> dict:
        self.ensure_loaded()
        res = self.predict_with_probabilities(data)
        return {
            "prediction_code": res["prediction_code"],
            "risk_status": res["prediction_label"],
            "message": "Dự đoán thành công",
        }

project_risk_service = ProjectRiskService()
