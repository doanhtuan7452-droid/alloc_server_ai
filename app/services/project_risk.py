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
        if isinstance(data, dict):
            methodology = data.get("methodology")
            if not methodology:
                # Infer from binary fields
                if data.get("Methodology_Used_Hybrid") == 1:
                    methodology = "hybrid"
                elif data.get("Methodology_Used_Kanban") == 1:
                    methodology = "kanban"
                elif data.get("Methodology_Used_Scrum") == 1:
                    methodology = "scrum"
                elif data.get("Methodology_Used_Waterfall") == 1:
                    methodology = "waterfall"
                else:
                    methodology = "agile"
            else:
                methodology = str(methodology).lower()
                
            p_days = data.get("Project_Duration_Days")
            exp_budget = data.get("Expected_Budget")
            t_size = data.get("Team_Size")
            avg_skill = data.get("Avg_Team_Skill_Level")
            comp_score = data.get("Complexity_Score")
            b_util = data.get("Budget_Utilization")
        else:
            methodology = data.methodology.lower() if data.methodology else None
            if not methodology:
                # Infer from binary fields
                if data.Methodology_Used_Hybrid == 1:
                    methodology = "hybrid"
                elif data.Methodology_Used_Kanban == 1:
                    methodology = "kanban"
                elif data.Methodology_Used_Scrum == 1:
                    methodology = "scrum"
                elif data.Methodology_Used_Waterfall == 1:
                    methodology = "waterfall"
                else:
                    methodology = "agile"
                    
            p_days = data.Project_Duration_Days
            exp_budget = data.Expected_Budget
            t_size = data.Team_Size
            avg_skill = data.Avg_Team_Skill_Level
            comp_score = data.Complexity_Score
            b_util = data.Budget_Utilization

        input_dict = {
            "Project_Duration_Days": p_days,
            "Expected_Budget": exp_budget,
            "Team_Size": t_size,
            "Avg_Team_Skill_Level": avg_skill,
            "Complexity_Score": comp_score,
            "Budget_Utilization": b_util,
            "Methodology_Used_Hybrid": 1 if methodology == "hybrid" else 0,
            "Methodology_Used_Kanban": 1 if methodology == "kanban" else 0,
            "Methodology_Used_Scrum": 1 if methodology == "scrum" else 0,
            "Methodology_Used_Waterfall": 1 if methodology == "waterfall" else 0
        }
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
