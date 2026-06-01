from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(
    title="Employee Suggestion AI API",
    description="API for C# server integration with employee allocation and project risk models.",
    version="1.0",
)

BASE_DIR = Path(__file__).resolve().parent

EMPLOYEE_MODEL_DIR = BASE_DIR / "models" / "employee_recommendation"
EMPLOYEE_MODEL_PATH = EMPLOYEE_MODEL_DIR / "hr_allocation_ai_model.pkl"
EMPLOYEE_SCALER_PATH = EMPLOYEE_MODEL_DIR / "hr_scaler.pkl"
EMPLOYEE_LABEL_ENCODER_PATH = EMPLOYEE_MODEL_DIR / "hr_label_encoder.pkl"

PROJECT_RISK_MODEL_DIR = BASE_DIR / "models" / "project_risk"
PROJECT_RISK_MODEL_PATH = PROJECT_RISK_MODEL_DIR / "project_risk_model.pkl"
PROJECT_RISK_SCALER_PATH = PROJECT_RISK_MODEL_DIR / "project_scaler.pkl"

ALLOCATION_FEATURE_COLUMNS = [
    "experience_years",
    "education_level",
    "skill_level",
    "technical_skill_score",
    "communication_score",
    "leadership_score",
    "problem_solving_score",
    "task_complexity",
    "required_skill_level",
    "deadline_days",
    "workload_hours",
    "task_priority",
    "team_size",
    "attendance_rate",
    "performance_rating",
    "conflict_rate",
    "skill_gap",
    "hours_per_day",
    "avg_soft_skill",
]

PROJECT_RISK_FEATURE_COLUMNS = [
    "Project_Duration_Days",
    "Expected_Budget",
    "Team_Size",
    "Avg_Team_Skill_Level",
    "Complexity_Score",
    "Budget_Utilization",
    "Methodology_Used_Hybrid",
    "Methodology_Used_Kanban",
    "Methodology_Used_Scrum",
    "Methodology_Used_Waterfall",
]


def model_to_dict(data: BaseModel) -> dict:
    if hasattr(data, "model_dump"):
        return data.model_dump()
    return data.dict()


def load_joblib(path: Path, name: str):
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"[!] Could not load {name} from {path}: {exc}")
        return None


employee_model = load_joblib(EMPLOYEE_MODEL_PATH, "employee allocation model")
employee_scaler = load_joblib(EMPLOYEE_SCALER_PATH, "employee allocation scaler")
employee_label_encoder = load_joblib(
    EMPLOYEE_LABEL_ENCODER_PATH,
    "employee allocation label encoder",
)

project_risk_model = load_joblib(PROJECT_RISK_MODEL_PATH, "project risk model")
project_risk_scaler = load_joblib(PROJECT_RISK_SCALER_PATH, "project risk scaler")


class AllocationRequest(BaseModel):
    experience_years: float
    education_level: str
    skill_level: str
    technical_skill_score: float
    communication_score: float
    leadership_score: float
    problem_solving_score: float
    task_complexity: str
    required_skill_level: str
    deadline_days: int
    workload_hours: float
    task_priority: str
    team_size: int
    attendance_rate: float
    performance_rating: str
    conflict_rate: float


class ProjectData(BaseModel):
    Project_Duration_Days: int
    Expected_Budget: float
    Team_Size: int
    Avg_Team_Skill_Level: float
    Complexity_Score: float
    Budget_Utilization: float
    Methodology_Used_Kanban: int
    Methodology_Used_Scrum: int
    Methodology_Used_Waterfall: int
    Methodology_Used_Hybrid: int


def ensure_employee_models_loaded() -> None:
    if (
        employee_model is None
        or employee_scaler is None
        or employee_label_encoder is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Employee recommendation model is not loaded.",
        )


def ensure_project_risk_models_loaded() -> None:
    if project_risk_model is None or project_risk_scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Project risk model is not loaded.",
        )


@app.post("/api/v1/suggest-allocation")
async def suggest_allocation(request: AllocationRequest):
    ensure_employee_models_loaded()

    try:
        input_dict = model_to_dict(request)
        df = pd.DataFrame([input_dict])

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

        df["skill_gap"] = df["skill_level"] - df["required_skill_level"]
        df["hours_per_day"] = df["workload_hours"] / (df["deadline_days"] + 1e-5)
        df["avg_soft_skill"] = df[
            ["communication_score", "leadership_score", "problem_solving_score"]
        ].mean(axis=1)

        df = df[ALLOCATION_FEATURE_COLUMNS]

        x_scaled = employee_scaler.transform(df)
        x_scaled_df = pd.DataFrame(x_scaled, columns=ALLOCATION_FEATURE_COLUMNS)
        predicted_class_idx = employee_model.predict(x_scaled_df)[0]
        predicted_status = employee_label_encoder.inverse_transform(
            [predicted_class_idx]
        )[0]

        return {
            "is_success": True,
            "prediction": predicted_status,
            "prediction_code": int(predicted_class_idx),
            "analyzed_metrics": {
                "skill_gap": float(df["skill_gap"].iloc[0]),
                "hours_per_day": float(df["hours_per_day"].iloc[0]),
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/predict-risk")
def predict_project_risk(data: ProjectData):
    ensure_project_risk_models_loaded()

    try:
        input_df = pd.DataFrame([model_to_dict(data)])
        input_df = input_df[PROJECT_RISK_FEATURE_COLUMNS]

        input_scaled = project_risk_scaler.transform(input_df)
        prediction = project_risk_model.predict(input_scaled)[0]
        risk_status = "High/Critical Risk" if prediction == 1 else "Low/Medium Risk"

        return {
            "prediction_code": int(prediction),
            "risk_status": risk_status,
            "message": "Dự đoán thành công",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
