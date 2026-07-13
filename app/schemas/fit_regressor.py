from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    experience_years: int = Field(..., example=5, description="Số năm kinh nghiệm làm việc")
    education_level: str = Field(..., example="Bachelor", description="Trình độ học vấn: High School, Diploma, Bachelor, Master, PhD")
    skill_level: str = Field(..., example="High", description="Cấp độ kỹ năng thực tế của nhân sự: Low, Medium, High, Expert")
    technical_skill_score: float = Field(..., example=82.5, description="Điểm đánh giá chuyên môn công nghệ (0-100)")
    communication_score: float = Field(..., example=78.0, description="Điểm đánh giá kỹ năng giao tiếp (0-100)")
    leadership_score: float = Field(..., example=60.0, description="Điểm đánh giá năng lực lãnh đạo (0-100)")
    problem_solving_score: float = Field(..., example=75.0, description="Điểm tư duy giải quyết vấn đề (0-100)")
    task_complexity: str = Field(..., example="Medium", description="Độ phức tạp của công việc: Low, Medium, High, Critical")
    required_skill_level: str = Field(..., example="Medium", description="Yêu cầu kỹ năng tối thiểu của task: Low, Medium, High, Expert")
    deadline_days: int = Field(..., example=12, description="Số ngày được giao để hoàn thành công việc")
    workload_hours: int = Field(..., example=45, description="Tổng số giờ công ước tính cho công việc")
    task_priority: str = Field(..., example="High", description="Mức độ ưu tiên của công việc: Low, Medium, High, Critical")
    team_size: int = Field(..., example=4, description="Số lượng thành viên phối hợp thực hiện task")
    attendance_rate: float = Field(..., example=96.5, description="Tỷ lệ chuyên cần/chấm công của nhân viên (0-100)")
    performance_rating: str = Field(..., example="Excellent", description="Đánh giá hiệu suất chu kỳ cũ: Poor, Average, Excellent, Outstanding")
    conflict_rate: float = Field(..., example=8.0, description="Tỷ lệ xung đột/nhật ký tranh chấp của nhân viên (0-100)")


class PredictionResponse(BaseModel):
    fit_percentage: float = Field(..., description="Điểm số mức độ phù hợp dự đoán dạng liên tục (0.0 - 100.0)")
    recommendation_status: str = Field(..., description="Khuyến nghị điều phối nhân sự tự động cho bộ phận HR")
