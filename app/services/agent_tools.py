from langchain_core.tools import StructuredTool
from app.schemas.allocation import AllocationRequest
from app.schemas.project_risk import ProjectData
from app.schemas.assessment import PersonnelAssessmentRequest, ProjectRiskAssessmentRequest
from app.services.employee import employee_service
from app.services.project_risk import project_risk_service
from app.services.allocation_assessment import allocation_assessment_service
from app.services.project_risk_assessment import project_risk_assessment_service

# Legacy wrappers
def predict_employee_allocation_wrapper(**kwargs) -> dict:
    """Wrapper function to instantiate AllocationRequest and call the ML service."""
    request_obj = AllocationRequest(**kwargs)
    return employee_service.predict(request_obj)

def predict_project_risk_wrapper(**kwargs) -> dict:
    """Wrapper function to instantiate ProjectData and call the ML service."""
    data_obj = ProjectData(**kwargs)
    return project_risk_service.predict(data_obj)

# New async wrappers for deterministic orchestrator execution (bypassing LLM explanations)
async def assess_employee_allocation_wrapper(**kwargs) -> dict:
    """Wrapper function to call the allocation assessment service asynchronously, bypassing LLM explanation."""
    request_obj = PersonnelAssessmentRequest(**kwargs)
    res = await allocation_assessment_service.assess(request_obj, bypass_llm=True)
    return res.model_dump()

async def assess_project_risk_wrapper(**kwargs) -> dict:
    """Wrapper function to call the project risk assessment service asynchronously, bypassing LLM explanation."""
    request_obj = ProjectRiskAssessmentRequest(**kwargs)
    res = await project_risk_assessment_service.assess(request_obj, bypass_llm=True)
    return res.model_dump()

# Create StructuredTools
predict_employee_allocation_tool = StructuredTool.from_function(
    func=predict_employee_allocation_wrapper,
    name="predict_employee_allocation",
    description=(
        "Sử dụng công cụ này để dự báo sự phù hợp và khả năng thành công của một nhân viên "
        "khi được phân công vào một công việc cụ thể. Đầu vào bao gồm thông tin chi tiết về "
        "kinh nghiệm, học vấn, kỹ năng của nhân viên cũng như độ phức tạp, thời hạn của công việc."
    ),
    args_schema=AllocationRequest
)

predict_project_risk_tool = StructuredTool.from_function(
    func=predict_project_risk_wrapper,
    name="predict_project_risk",
    description=(
        "Sử dụng công cụ này để dự báo mức độ rủi ro của dự án (Thấp/Trung bình hoặc Cao/Nghiêm trọng) "
        "dựa trên các chỉ số tài chính, tiến độ, số lượng thành viên đội ngũ và phương pháp quản lý dự án."
    ),
    args_schema=ProjectData
)

assess_employee_allocation_tool = StructuredTool.from_function(
    coroutine=assess_employee_allocation_wrapper,
    name="assess_employee_allocation",
    description=(
        "Sử dụng công cụ này để đánh giá sự phù hợp nghiệp vụ và rủi ro chi tiết của nhân viên "
        "khi được phân công vào một công việc cụ thể. Trả về kết quả phân tích ML và các yếu tố "
        "thành công/thách thức nghiệp vụ (không bao gồm giải thích ngôn ngữ tự nhiên từ LLM)."
    ),
    args_schema=PersonnelAssessmentRequest
)

assess_project_risk_tool = StructuredTool.from_function(
    coroutine=assess_project_risk_wrapper,
    name="assess_project_risk",
    description=(
        "Sử dụng công cụ này để đánh giá mức độ rủi ro chi tiết của dự án dựa trên tiến độ, ngân sách, "
        "quy mô đội ngũ và phương pháp quản lý. Trả về kết quả phân tích ML và các cảnh báo nghiệp vụ "
        "(không bao gồm giải thích ngôn ngữ tự nhiên từ LLM)."
    ),
    args_schema=ProjectRiskAssessmentRequest
)

# Export the tools list for the agent
agent_tools = [
    predict_employee_allocation_tool,
    predict_project_risk_tool,
    assess_employee_allocation_tool,
    assess_project_risk_tool
]
