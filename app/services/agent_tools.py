from langchain_core.tools import StructuredTool
from app.schemas.allocation import AllocationRequest
from app.schemas.project_risk import ProjectData
from app.schemas.assessment import PersonnelAssessmentRequest, ProjectRiskAssessmentRequest
from app.services.predict_strategies import PredictStrategyFactory
from app.services.allocation_assessment import allocation_assessment_service
from app.services.project_risk_assessment import project_risk_assessment_service
from fastapi.concurrency import run_in_threadpool
from app.services.tool_safety_guard import tool_safety_guard

# Async wrappers using Strategy Factory to offload synchronous code to Starlette threadpool
async def predict_employee_allocation_wrapper(**kwargs) -> dict:
    """Wrapper function to instantiate AllocationRequest and call the ML service on threadpool."""
    is_safe, safety_err = tool_safety_guard.validate_tool_call("predict_employee_allocation", kwargs)
    if not is_safe:
        return {"error": f"Lỗi an toàn: {safety_err}"}
    request_obj = AllocationRequest(**kwargs)
    strategy = PredictStrategyFactory.get_strategy("allocation")
    return await strategy.predict(request_obj)

async def predict_project_risk_wrapper(**kwargs) -> dict:
    """Wrapper function to instantiate ProjectData and call the ML service on threadpool."""
    is_safe, safety_err = tool_safety_guard.validate_tool_call("predict_project_risk", kwargs)
    if not is_safe:
        return {"error": f"Lỗi an toàn: {safety_err}"}
    data_obj = ProjectData(**kwargs)
    strategy = PredictStrategyFactory.get_strategy("project_risk")
    return await strategy.predict(data_obj)

# Async wrappers for deterministic orchestrator execution (bypassing LLM explanations)
async def assess_employee_allocation_wrapper(**kwargs) -> dict:
    """Wrapper function to call the allocation assessment service asynchronously on threadpool, bypassing LLM explanation."""
    is_safe, safety_err = tool_safety_guard.validate_tool_call("assess_employee_allocation", kwargs)
    if not is_safe:
        return {"error": f"Lỗi an toàn: {safety_err}"}
    request_obj = PersonnelAssessmentRequest(**kwargs)
    # assess performs async operations, but underlying ML predictions are offloaded
    res = await allocation_assessment_service.assess(request_obj, bypass_llm=True)
    return res.model_dump()

async def assess_project_risk_wrapper(**kwargs) -> dict:
    """Wrapper function to call the project risk assessment service asynchronously on threadpool, bypassing LLM explanation."""
    is_safe, safety_err = tool_safety_guard.validate_tool_call("assess_project_risk", kwargs)
    if not is_safe:
        return {"error": f"Lỗi an toàn: {safety_err}"}
    request_obj = ProjectRiskAssessmentRequest(**kwargs)
    res = await project_risk_assessment_service.assess(request_obj, bypass_llm=True)
    return res.model_dump()

# Create StructuredTools using the asynchronous wrappers
predict_employee_allocation_tool = StructuredTool.from_function(
    coroutine=predict_employee_allocation_wrapper,
    name="predict_employee_allocation",
    description=(
        "Sử dụng công cụ này để dự báo sự phù hợp và khả năng thành công của một nhân viên "
        "khi được phân công vào một công việc cụ thể. Đầu vào bao gồm thông tin chi tiết về "
        "kinh nghiệm, học vấn, kỹ năng của nhân viên cũng như độ phức tạp, thời hạn của công việc."
    ),
    args_schema=AllocationRequest
)

predict_project_risk_tool = StructuredTool.from_function(
    coroutine=predict_project_risk_wrapper,
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
