import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load env variables at the very beginning of configuration initialization
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

class Settings:
    # Root directory of the project
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # Employee Recommendation model paths
    EMPLOYEE_MODEL_DIR: Path = BASE_DIR / "model_ai" / "employee_recommendation"
    EMPLOYEE_MODEL_PATH: Path = EMPLOYEE_MODEL_DIR / "hr_allocation_ai_model.pkl"
    EMPLOYEE_SCALER_PATH: Path = EMPLOYEE_MODEL_DIR / "hr_scaler.pkl"
    EMPLOYEE_LABEL_ENCODER_PATH: Path = EMPLOYEE_MODEL_DIR / "hr_label_encoder.pkl"
    
    # Project Risk model paths
    PROJECT_RISK_MODEL_DIR: Path = BASE_DIR / "model_ai" / "project_risk"
    PROJECT_RISK_MODEL_PATH: Path = PROJECT_RISK_MODEL_DIR / "project_risk_model.pkl"
    PROJECT_RISK_SCALER_PATH: Path = PROJECT_RISK_MODEL_DIR / "project_scaler.pkl"
    
    # Feature columns used for employee suggestion mapping
    ALLOCATION_FEATURE_COLUMNS: List[str] = [
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
    
    # Feature columns used for project risk prediction mapping
    PROJECT_RISK_FEATURE_COLUMNS: List[str] = [
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

    # LLM & Agent Configurations
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # API Keys & Endpoints
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

settings = Settings()
