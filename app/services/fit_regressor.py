import os
import joblib
import pandas as pd
import numpy as np
from fastapi import HTTPException
from app.core.config import settings
from app.schemas.fit_regressor import PredictionRequest, PredictionResponse

class FitRegressorService:
    def __init__(self):
        self.model = None
        self.scaler = None

    def load_models(self):
        """Loads the Random Forest Regressor and Scaler models into memory."""
        if self.model is not None and self.scaler is not None:
            return
        
        try:
            if os.path.exists(settings.FIT_REGRESSOR_MODEL_PATH) and os.path.exists(settings.FIT_REGRESSOR_SCALER_PATH):
                self.model = joblib.load(settings.FIT_REGRESSOR_MODEL_PATH)
                self.scaler = joblib.load(settings.FIT_REGRESSOR_SCALER_PATH)
                print("[+] Đã tải thành công Fit Regressor Model và Scaler phục vụ dự đoán trực tuyến.")
            else:
                print("[!] Không tìm thấy file model hoặc bộ chuẩn hóa trong thư mục 'model_ai/fit_regressor/'.")
        except Exception as exc:
            print(f"[!] Lỗi nghiêm trọng khi khởi tạo Fit Regressor Model: {str(exc)}")

    def ensure_loaded(self):
        """Ensures the models are loaded, otherwise raises HTTPException 503."""
        if self.model is None or self.scaler is None:
            raise HTTPException(
                status_code=503,
                detail="Hệ thống AI đang bảo trì: Không tìm thấy tệp trọng số mô hình hoặc bộ cấu hình scaler của Fit Regressor."
            )

    def prepare_batch_features(self, requests: list) -> pd.DataFrame:
        """Convert a batch of requests into a single processed DataFrame for regression, securing all inputs."""
        input_dicts = []
        for req in requests:
            if hasattr(req, "model_dump"):
                input_dicts.append(req.model_dump())
            elif hasattr(req, "dict"):
                input_dicts.append(req.dict())
            elif isinstance(req, dict):
                input_dicts.append(req)
            else:
                input_dicts.append(dict(req))
        
        df = pd.DataFrame(input_dicts)

        # Standard default fallbacks for missing/NaN values
        defaults = {
            "education_level": "bachelor",
            "performance_rating": "average",
            "task_complexity": "medium",
            "required_skill_level": "medium",
            "task_priority": "medium",
            "skill_level": "medium",
            "experience_years": 0.0,
            "technical_skill_score": 50.0,
            "communication_score": 50.0,
            "leadership_score": 50.0,
            "problem_solving_score": 50.0,
            "deadline_days": 10,
            "workload_hours": 40.0,
            "team_size": 3,
            "attendance_rate": 95.0,
            "conflict_rate": 5.0
        }

        for col, val in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
            else:
                df[col] = val

        # Value maps preserved 1:1
        edu_map = {'high school': 0, 'diploma': 1, 'bachelor': 2, 'master': 3, 'phd': 4}
        perf_map = {'poor': 0, 'average': 1, 'good': 2, 'excellent': 3, 'outstanding': 4}
        complex_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        prio_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        skill_map = {'low': 0, 'medium': 1, 'high': 2, 'expert': 3}

        def clean_string_series(series: pd.Series) -> pd.Series:
            return series.astype(str).str.lower().str.strip()

        try:
            df["education_level"] = clean_string_series(df["education_level"]).map(edu_map).fillna(2)
            df["performance_rating"] = clean_string_series(df["performance_rating"]).map(perf_map).fillna(1)
            df["task_complexity"] = clean_string_series(df["task_complexity"]).map(complex_map).fillna(1)
            df["required_skill_level"] = clean_string_series(df["required_skill_level"]).map(skill_map).fillna(1)
            df["task_priority"] = clean_string_series(df["task_priority"]).map(prio_map).fillna(1)
            df["skill_level"] = clean_string_series(df["skill_level"]).map(skill_map).fillna(1)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Lỗi phân rã định dạng nhãn chữ theo lô (Batch categorical mapping): {str(e)}"
            )

        # Value Clipping to prevent bias/skewness and handle invalid numbers
        score_columns = [
            "technical_skill_score",
            "communication_score",
            "leadership_score",
            "problem_solving_score",
            "attendance_rate",
            "conflict_rate",
        ]
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(50.0).clip(lower=0.0, upper=100.0)
            
        df["experience_years"] = pd.to_numeric(df["experience_years"], errors='coerce').fillna(0.0).clip(lower=0.0)
        df["deadline_days"] = pd.to_numeric(df["deadline_days"], errors='coerce').fillna(10).astype(int)
        df["workload_hours"] = pd.to_numeric(df["workload_hours"], errors='coerce').fillna(40.0)
        df["team_size"] = pd.to_numeric(df["team_size"], errors='coerce').fillna(3).astype(int)

        # Derived features engineering
        df["hours_per_day"] = df["workload_hours"] / (df["deadline_days"] + 1e-5)
        df["skill_gap"] = df["skill_level"] - df["required_skill_level"]
        df["avg_soft_skill"] = df[
            ["communication_score", "leadership_score", "problem_solving_score"]
        ].mean(axis=1)

        # Ensure column order matches settings configuration precisely
        return df[settings.FIT_REGRESSOR_FEATURE_COLUMNS]

    def prepare_features(self, request: PredictionRequest) -> pd.DataFrame:
        """Applies exact categorical mappings and derived feature calculations as trained."""
        return self.prepare_batch_features([request])

    def predict_batch(self, requests: list) -> list:
        """Executes regression inference workflow on a batch of requests and returns fit percentages."""
        self.ensure_loaded()
        if not requests:
            return []
        
        # 1. Feature Preprocessing
        df_features = self.prepare_batch_features(requests)

        # 2. Scaler transformation
        try:
            scaled_array = self.scaler.transform(df_features)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Lỗi Runtime Preprocessing theo lô (Batch Scaler Transform): {str(e)}"
            )

        # 3. Model inference and clamping
        try:
            scaled_df = pd.DataFrame(scaled_array, columns=settings.FIT_REGRESSOR_FEATURE_COLUMNS)
            raw_preds = self.model.predict(scaled_df)
            fit_percentages = np.clip(raw_preds, 0.0, 100.0)
            return [round(float(val), 2) for val in fit_percentages]
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Lỗi trong quá trình suy luận mô hình theo lô (Batch Model Inference): {str(e)}"
            )

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Executes full regression inference workflow and maps result to recommendation status."""
        self.ensure_loaded()
        
        # 1. Feature Preprocessing
        df_features = self.prepare_features(request)

        # 2. Scaler transformation
        try:
            scaled_array = self.scaler.transform(df_features)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Lỗi Runtime Preprocessing (Scaler Transform): {str(e)}"
            )

        # 3. Model inference and clamping
        try:
            scaled_df = pd.DataFrame(scaled_array, columns=settings.FIT_REGRESSOR_FEATURE_COLUMNS)
            raw_pred = self.model.predict(scaled_df)
            fit_percentage = float(np.clip(raw_pred[0], 0.0, 100.0))
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Lỗi trong quá trình suy luận mô hình (Model Inference): {str(e)}"
            )

        # 4. Threshold recommendation rules mapping
        if fit_percentage >= 85.0:
            recommendation = "Optimal Allocation (Phân bổ xuất sắc - Khuyến khích giao việc ngay)"
        elif fit_percentage >= 65.0:
            recommendation = "Suitable Allocation (Phân bổ phù hợp - Nhân sự đủ năng lực đáp ứng)"
        elif fit_percentage >= 45.0:
            recommendation = "Suboptimal Allocation (Kém tối ưu - Cần có Tech Lead/Quản lý kèm cặp sát sao)"
        else:
            recommendation = "Reassignment Required (Không phù hợp - Quá tải hoặc hổng kiến thức nghiêm trọng, bắt buộc đổi người)"

        return PredictionResponse(
            fit_percentage=round(fit_percentage, 2),
            recommendation_status=recommendation
        )

# Singleton Service Instance
fit_regressor_service = FitRegressorService()
