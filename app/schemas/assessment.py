from typing import Optional, Literal, List, Dict, Union
from pydantic import BaseModel, Field
from app.schemas.allocation import AllocationRequest
from app.schemas.project_risk import ProjectData
from app.schemas.mongo_chat import TokenUsage

class AdapterResult(BaseModel):
    allocation_request: AllocationRequest
    assumptions: Dict[str, Union[float, str, int]]
    missing_fields: List[str]
    confidence_penalty: float

class PersonnelAssessmentRequest(BaseModel):
    request_type: Literal["single"] = "single"
    experience_years: float = Field(ge=0)
    skill_level: Literal["low", "medium", "high", "expert"]
    technical_skill_score: float = Field(ge=0.0, le=100.0)
    communication_score: float = Field(ge=0.0, le=100.0)
    task_complexity: Literal["low", "medium", "high", "critical"]
    deadline_days: int = Field(gt=0)
    
    # Optional raw fields from AllocationRequest (with defaults handled in adapter)
    education_level: Optional[str] = None
    leadership_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    problem_solving_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    required_skill_level: Optional[str] = None
    workload_hours: Optional[float] = Field(None, ge=0.0)
    task_priority: Optional[str] = None
    team_size: Optional[int] = Field(None, ge=0)
    attendance_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    performance_rating: Optional[str] = None
    conflict_rate: Optional[float] = Field(None, ge=0.0, le=100.0)

    # Optional LLM settings
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None

    def to_allocation_request(self) -> AdapterResult:
        # Define default assumptions for missing optional raw fields
        defaults = {
            "education_level": "bachelor",
            "leadership_score": 50.0,
            "problem_solving_score": 50.0,
            "required_skill_level": "medium",
            "workload_hours": 40.0,
            "task_priority": "medium",
            "team_size": 3,
            "attendance_rate": 95.0,
            "performance_rating": "average",
            "conflict_rate": 5.0
        }

        assumptions = {}
        missing_fields = []

        # Check which optional fields were not provided by client
        for field, default_val in defaults.items():
            user_val = getattr(self, field, None)
            if user_val is None:
                assumptions[field] = default_val
                missing_fields.append(field)

        # Calculate dynamic confidence penalty (2.5 penalty per missing field)
        confidence_penalty = float(len(missing_fields) * 2.5)

        allocation_req = AllocationRequest(
            experience_years=float(self.experience_years),
            education_level=self.education_level if self.education_level is not None else defaults["education_level"],
            skill_level=self.skill_level,
            technical_skill_score=self.technical_skill_score,
            communication_score=self.communication_score,
            leadership_score=self.leadership_score if self.leadership_score is not None else defaults["leadership_score"],
            problem_solving_score=self.problem_solving_score if self.problem_solving_score is not None else defaults["problem_solving_score"],
            task_complexity=self.task_complexity,
            required_skill_level=self.required_skill_level if self.required_skill_level is not None else defaults["required_skill_level"],
            deadline_days=self.deadline_days,
            workload_hours=self.workload_hours if self.workload_hours is not None else defaults["workload_hours"],
            task_priority=self.task_priority if self.task_priority is not None else defaults["task_priority"],
            team_size=self.team_size if self.team_size is not None else defaults["team_size"],
            attendance_rate=self.attendance_rate if self.attendance_rate is not None else defaults["attendance_rate"],
            performance_rating=self.performance_rating if self.performance_rating is not None else defaults["performance_rating"],
            conflict_rate=self.conflict_rate if self.conflict_rate is not None else defaults["conflict_rate"]
        )

        return AdapterResult(
            allocation_request=allocation_req,
            assumptions=assumptions,
            missing_fields=missing_fields,
            confidence_penalty=confidence_penalty
        )

class EmployeeAssessmentInput(BaseModel):
    employee_id: str = Field(..., description="Mã nhân viên")
    employee_name: Optional[str] = Field(None, description="Tên nhân viên")
    experience_years: float = Field(ge=0)
    skill_level: Literal["low", "medium", "high", "expert"]
    technical_skill_score: float = Field(ge=0.0, le=100.0)
    communication_score: float = Field(ge=0.0, le=100.0)
    
    # Optional raw fields
    education_level: Optional[str] = None
    leadership_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    problem_solving_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    attendance_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    performance_rating: Optional[str] = None
    conflict_rate: Optional[float] = Field(None, ge=0.0, le=100.0)

    def to_allocation_request(
        self,
        task_complexity: str,
        deadline_days: int,
        required_skill_level: Optional[str] = None,
        workload_hours: Optional[float] = None,
        task_priority: Optional[str] = None,
        team_size: Optional[int] = None
    ) -> AdapterResult:
        defaults = {
            "education_level": "bachelor",
            "leadership_score": 50.0,
            "problem_solving_score": 50.0,
            "required_skill_level": "medium",
            "workload_hours": 40.0,
            "task_priority": "medium",
            "team_size": 3,
            "attendance_rate": 95.0,
            "performance_rating": "average",
            "conflict_rate": 5.0
        }

        assumptions = {}
        missing_fields = []

        # Construct candidate-level and task-level overrides
        candidate_vals = {
            "education_level": self.education_level,
            "leadership_score": self.leadership_score,
            "problem_solving_score": self.problem_solving_score,
            "attendance_rate": self.attendance_rate,
            "performance_rating": self.performance_rating,
            "conflict_rate": self.conflict_rate,
        }
        task_vals = {
            "required_skill_level": required_skill_level,
            "workload_hours": workload_hours,
            "task_priority": task_priority,
            "team_size": team_size,
        }

        for field, default_val in defaults.items():
            if field in candidate_vals:
                val = candidate_vals[field]
            else:
                val = task_vals[field]
            
            if val is None:
                assumptions[field] = default_val
                missing_fields.append(field)

        confidence_penalty = float(len(missing_fields) * 2.5)

        allocation_req = AllocationRequest(
            experience_years=float(self.experience_years),
            education_level=self.education_level if self.education_level is not None else defaults["education_level"],
            skill_level=self.skill_level,
            technical_skill_score=self.technical_skill_score,
            communication_score=self.communication_score,
            leadership_score=self.leadership_score if self.leadership_score is not None else defaults["leadership_score"],
            problem_solving_score=self.problem_solving_score if self.problem_solving_score is not None else defaults["problem_solving_score"],
            task_complexity=task_complexity,
            required_skill_level=required_skill_level if required_skill_level is not None else defaults["required_skill_level"],
            deadline_days=deadline_days,
            workload_hours=workload_hours if workload_hours is not None else defaults["workload_hours"],
            task_priority=task_priority if task_priority is not None else defaults["task_priority"],
            team_size=team_size if team_size is not None else defaults["team_size"],
            attendance_rate=self.attendance_rate if self.attendance_rate is not None else defaults["attendance_rate"],
            performance_rating=self.performance_rating if self.performance_rating is not None else defaults["performance_rating"],
            conflict_rate=self.conflict_rate if self.conflict_rate is not None else defaults["conflict_rate"]
        )

        return AdapterResult(
            allocation_request=allocation_req,
            assumptions=assumptions,
            missing_fields=missing_fields,
            confidence_penalty=confidence_penalty
        )

class BulkAssessmentRequest(BaseModel):
    request_type: Literal["bulk"] = "bulk"
    task_complexity: Literal["low", "medium", "high", "critical"]
    deadline_days: int = Field(gt=0)
    required_skill_level: Optional[str] = None
    workload_hours: Optional[float] = Field(None, ge=0.0)
    task_priority: Optional[str] = None
    team_size: Optional[int] = Field(None, ge=0)
    
    employees: List[EmployeeAssessmentInput] = Field(..., min_length=1)

    # Optional LLM settings
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None

class ProjectRiskAssessmentRequest(ProjectData):
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None

class AllocationAssessmentResponse(BaseModel):
    prediction_label: str
    prediction_code: int
    class_probabilities: Dict[str, float]
    confidence_score: float
    business_status_code: str
    business_status_text: str
    success_factors: List[str]
    potential_challenges: List[str]
    llm_insight: str
    explanation_source: str
    assumptions: Dict[str, Union[float, str, int]]
    missing_fields: List[str]
    confidence_penalty: float
    fit_percentage: float
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Báo cáo số token tiêu thụ")

class EmployeeAllocationAssessmentResult(BaseModel):
    employee_id: str
    employee_name: Optional[str] = None
    fit_percentage: float = Field(ge=0.0, le=100.0)
    prediction_label: str
    prediction_code: int
    class_probabilities: Dict[str, float]
    confidence_score: float
    business_status_code: str
    business_status_text: str
    success_factors: List[str]
    potential_challenges: List[str]
    llm_insight: str
    explanation_source: str
    assumptions: Dict[str, Union[float, str, int]]
    missing_fields: List[str]
    confidence_penalty: float
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Báo cáo số token tiêu thụ cho ứng viên này")

class BulkAllocationAssessmentResponse(BaseModel):
    results: List[EmployeeAllocationAssessmentResult]
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Tổng số token tiêu thụ cho toàn bộ lượt đánh giá")

class ProjectRiskAssessmentResponse(BaseModel):
    prediction_label: str
    prediction_code: int
    class_probabilities: Dict[str, float]
    confidence_score: float
    business_status_code: str
    business_status_text: str
    success_factors: List[str]
    potential_challenges: List[str]
    llm_insight: str
    explanation_source: str
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Báo cáo số token tiêu thụ")
