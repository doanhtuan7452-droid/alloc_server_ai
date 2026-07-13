import joblib
import pandas as pd
from fastapi import HTTPException
from app.core.config import settings
from app.schemas.allocation import AllocationRequest

class EmployeeRecommendationService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None

    def load_models(self):
        """Load the ML pipeline components into memory. Executed at startup."""
        if self.model is not None and self.scaler is not None and self.label_encoder is not None:
            return
        try:
            self.model = joblib.load(settings.EMPLOYEE_MODEL_PATH)
            self.scaler = joblib.load(settings.EMPLOYEE_SCALER_PATH)
            self.label_encoder = joblib.load(settings.EMPLOYEE_LABEL_ENCODER_PATH)
            print("[+] Employee recommendation models loaded successfully.")
        except Exception as exc:
            print(f"[!] Could not load employee recommendation models: {exc}")

    def ensure_loaded(self):
        if self.model is None or self.scaler is None or self.label_encoder is None:
            raise HTTPException(
                status_code=503,
                detail="Employee recommendation model is not loaded.",
            )

    def prepare_features(self, request: AllocationRequest) -> pd.DataFrame:
        """Convert request into the scaled/processed dataframe containing all feature columns."""
        # Convert Pydantic request to dict and DataFrame
        input_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        df = pd.DataFrame([input_dict])

        # Value maps preserved 1:1
        edu_map = {
            "high school": 0,
            "diploma": 1,
            "bachelor": 2,
            "master": 3,
            "phd": 4,
        }
        perf_map = {"poor": 0, "average": 1, "excellent": 2, "outstanding": 3}
        complex_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        req_skill_map = {"low": 0, "medium": 1, "high": 2, "expert": 3}
        prio_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        skill_map = {"low": 0, "medium": 1, "high": 2, "expert": 3}

        df["education_level"] = (
            df["education_level"].str.lower().str.strip().map(edu_map).fillna(2)
        )
        df["performance_rating"] = (
            df["performance_rating"].str.lower().str.strip().map(perf_map).fillna(1)
        )
        df["task_complexity"] = (
            df["task_complexity"].str.lower().str.strip().map(complex_map).fillna(1)
        )
        df["required_skill_level"] = (
            df["required_skill_level"]
            .str.lower()
            .str.strip()
            .map(req_skill_map)
            .fillna(1)
        )
        df["task_priority"] = (
            df["task_priority"].str.lower().str.strip().map(prio_map).fillna(1)
        )
        df["skill_level"] = (
            df["skill_level"].str.lower().str.strip().map(skill_map).fillna(1)
        )

        score_columns = [
            "technical_skill_score",
            "communication_score",
            "leadership_score",
            "problem_solving_score",
            "attendance_rate",
            "conflict_rate",
        ]
        for col in score_columns:
            df[col] = df[col].clip(lower=0, upper=100)

        # Derived features
        df["skill_gap"] = df["skill_level"] - df["required_skill_level"]
        df["hours_per_day"] = df["workload_hours"] / (df["deadline_days"] + 1e-5)
        df["avg_soft_skill"] = df[
            ["communication_score", "leadership_score", "problem_solving_score"]
        ].mean(axis=1)

        # Filter columns to match feature columns list
        return df

    def prepare_batch_features(self, requests: list[AllocationRequest]) -> pd.DataFrame:
        """Convert a batch of requests into a single scaled/processed dataframe."""
        input_dicts = [req.model_dump() if hasattr(req, "model_dump") else req.dict() for req in requests]
        df = pd.DataFrame(input_dicts)

        edu_map = {
            "high school": 0,
            "diploma": 1,
            "bachelor": 2,
            "master": 3,
            "phd": 4,
        }
        perf_map = {"poor": 0, "average": 1, "excellent": 2, "outstanding": 3}
        complex_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        req_skill_map = {"low": 0, "medium": 1, "high": 2, "expert": 3}
        prio_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        skill_map = {"low": 0, "medium": 1, "high": 2, "expert": 3}

        df["education_level"] = (
            df["education_level"].str.lower().str.strip().map(edu_map).fillna(2)
        )
        df["performance_rating"] = (
            df["performance_rating"].str.lower().str.strip().map(perf_map).fillna(1)
        )
        df["task_complexity"] = (
            df["task_complexity"].str.lower().str.strip().map(complex_map).fillna(1)
        )
        df["required_skill_level"] = (
            df["required_skill_level"]
            .str.lower()
            .str.strip()
            .map(req_skill_map)
            .fillna(1)
        )
        df["task_priority"] = (
            df["task_priority"].str.lower().str.strip().map(prio_map).fillna(1)
        )
        df["skill_level"] = (
            df["skill_level"].str.lower().str.strip().map(skill_map).fillna(1)
        )

        score_columns = [
            "technical_skill_score",
            "communication_score",
            "leadership_score",
            "problem_solving_score",
            "attendance_rate",
            "conflict_rate",
        ]

        for col in score_columns:
            df[col] = df[col].clip(lower=0, upper=100)

        # Derived features
        df["skill_gap"] = df["skill_level"] - df["required_skill_level"]
        df["hours_per_day"] = df["workload_hours"] / (df["deadline_days"] + 1e-5)
        df["avg_soft_skill"] = df[
            ["communication_score", "leadership_score", "problem_solving_score"]
        ].mean(axis=1)

        return df

    def predict_batch_with_probabilities(self, requests: list[AllocationRequest]) -> list[dict]:
        """Runs batch inference and returns detailed predictions including raw class probabilities and confidence for all requests."""
        self.ensure_loaded()
        if not requests:
            return []
        
        df = self.prepare_batch_features(requests)
        df_filtered = df[settings.ALLOCATION_FEATURE_COLUMNS]

        # Scaler transformation and prediction
        x_scaled = self.scaler.transform(df_filtered)
        x_scaled_df = pd.DataFrame(x_scaled, columns=settings.ALLOCATION_FEATURE_COLUMNS)
        
        predicted_class_indices = self.model.predict(x_scaled_df)
        predicted_statuses = self.label_encoder.inverse_transform(predicted_class_indices)
        
        # Extract class probabilities (0.0 to 1.0 scale)
        probas = self.model.predict_proba(x_scaled_df)
        model_classes = self.model.classes_
        decoded_labels = self.label_encoder.inverse_transform(model_classes)
        
        results = []
        for i, (pred_idx, pred_status, proba) in enumerate(zip(predicted_class_indices, predicted_statuses, probas)):
            class_probabilities = {str(label): float(prob) for label, prob in zip(decoded_labels, proba)}
            
            # Calculate confidence score (0.0 to 100.0 scale) using text labels for absolute robustness
            if pred_status == "Optimal":
                confidence_score = class_probabilities.get("Optimal", 0.0) * 100.0
            else:
                confidence_score = (
                    class_probabilities.get("Reassignment Required", 0.0)
                    + class_probabilities.get("Suboptimal", 0.0)
                ) * 100.0

            results.append({
                "prediction_label": pred_status,
                "prediction_code": int(pred_idx),
                "class_probabilities": class_probabilities,
                "confidence_score": confidence_score,
                "analyzed_metrics": {
                    "skill_gap": float(df["skill_gap"].iloc[i]),
                    "hours_per_day": float(df["hours_per_day"].iloc[i]),
                },
            })
            
        return results

    def predict_with_probabilities(self, request: AllocationRequest) -> dict:
        """Runs inference and returns detailed predictions including raw class probabilities and confidence."""
        self.ensure_loaded()
        df = self.prepare_features(request)
        df_filtered = df[settings.ALLOCATION_FEATURE_COLUMNS]

        # Scaler transformation and prediction
        x_scaled = self.scaler.transform(df_filtered)
        x_scaled_df = pd.DataFrame(x_scaled, columns=settings.ALLOCATION_FEATURE_COLUMNS)
        
        predicted_class_idx = self.model.predict(x_scaled_df)[0]
        predicted_status = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Extract class probabilities (0.0 to 1.0 scale)
        proba = self.model.predict_proba(x_scaled_df)[0]
        model_classes = self.model.classes_
        decoded_labels = self.label_encoder.inverse_transform(model_classes)
        class_probabilities = {str(label): float(prob) for label, prob in zip(decoded_labels, proba)}
        
        # Calculate confidence score (0.0 to 100.0 scale) using text labels for absolute robustness
        if predicted_status == "Optimal":
            confidence_score = class_probabilities.get("Optimal", 0.0) * 100.0
        else:
            confidence_score = (
                class_probabilities.get("Reassignment Required", 0.0)
                + class_probabilities.get("Suboptimal", 0.0)
            ) * 100.0

        return {
            "prediction_label": predicted_status,
            "prediction_code": int(predicted_class_idx),
            "class_probabilities": class_probabilities,
            "confidence_score": confidence_score,
            "analyzed_metrics": {
                "skill_gap": float(df["skill_gap"].iloc[0]),
                "hours_per_day": float(df["hours_per_day"].iloc[0]),
            },
        }

    def predict(self, request: AllocationRequest) -> dict:
        self.ensure_loaded()
        res = self.predict_with_probabilities(request)
        return {
            "is_success": True,
            "prediction": res["prediction_label"],
            "prediction_code": res["prediction_code"],
            "analyzed_metrics": res["analyzed_metrics"],
        }

employee_service = EmployeeRecommendationService()
