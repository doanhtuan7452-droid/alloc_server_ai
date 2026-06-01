from pydantic import BaseModel

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
