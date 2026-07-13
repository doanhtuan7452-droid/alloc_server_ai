from fastapi import HTTPException
from langchain_core.messages import SystemMessage, HumanMessage
from app.schemas.assessment import ProjectRiskAssessmentRequest, ProjectRiskAssessmentResponse
from app.schemas.mongo_chat import TokenUsage
from app.services.project_risk import project_risk_service
from app.services.llm_factory import LLMFactory
from app.services.memory_service import count_tokens
from app.core.config import settings

class ProjectRiskAssessmentService:
    async def assess(self, request: ProjectRiskAssessmentRequest, bypass_llm: bool = False) -> ProjectRiskAssessmentResponse:
        # 1. Ensure project risk model is loaded
        project_risk_service.ensure_loaded()

        # 2. Predict ML probabilities
        # Wrap request to base ProjectData to match the ML service expectations
        ml_res = project_risk_service.predict_with_probabilities(request)

        # 3. Apply Business Rules Engine
        success_factors = []
        potential_challenges = []

        # Success Factors classification
        if request.Team_Size >= 5:
            success_factors.append("Đội ngũ nhân sự lớn (>= 5 thành viên)")
        if request.Avg_Team_Skill_Level >= 3.5:
            success_factors.append("Trình độ trung bình đội ngũ cao (>= 3.5)")
        if request.Complexity_Score <= 2.5:
            success_factors.append("Độ phức tạp công việc thấp đến trung bình (<= 2.5)")
        if request.Budget_Utilization <= 0.8:
            success_factors.append("Tỷ lệ sử dụng ngân sách tối ưu (<= 80%)")

        # Potential Challenges classification
        if request.Project_Duration_Days <= 30:
            potential_challenges.append("Thời gian thực hiện ngắn (<= 30 ngày)")
        if request.Complexity_Score >= 4.0:
            potential_challenges.append("Độ phức tạp công việc cao (>= 4.0)")
        if request.Budget_Utilization >= 1.0:
            potential_challenges.append("Vượt quá hoặc chạm ngưỡng ngân sách (>= 100%)")
        if request.Team_Size <= 2:
            potential_challenges.append("Đội ngũ quá ít người (<= 2 thành viên)")
        if request.Avg_Team_Skill_Level <= 2.0:
            potential_challenges.append("Trình độ trung bình đội ngũ thấp (<= 2.0)")

        # Severity Classification
        # - WARNING: ML predicts failure (code 1)
        # - REVIEW: ML predicts success (code 0) BUT potential challenges list is not empty
        # - APPROVED: ML predicts success (code 0) AND potential challenges list is empty
        if ml_res["prediction_code"] != 0:
            business_status_code = "WARNING"
            business_status_text = "CẢNH BÁO"
        elif potential_challenges:
            business_status_code = "REVIEW"
            business_status_text = "CẦN XEM XÉT"
        else:
            business_status_code = "APPROVED"
            business_status_text = "CHẤP THUẬN"

        # 4. LLM Explanation (or bypass)
        token_usage = TokenUsage()
        if bypass_llm:
            llm_insight = ""
            explanation_source = "bypassed"
        else:
            llm = LLMFactory.get_llm(
                provider=request.provider,
                model=request.model,
                temperature=request.temperature
            )

            system_prompt = (
                "Bạn là một trợ lý AI chuyên gia giải thích rủi ro dự án và quản lý tiến độ.\n"
                "Nhiệm vụ của bạn là giải thích kết quả đánh giá rủi ro dự án cho người dùng bằng tiếng Việt.\n"
                "RẤT QUAN TRỌNG:\n"
                "1. KHÔNG được thay đổi kết luận của mô hình ML và các luật nghiệp vụ. Kết luận đó là chân lý nền tảng (ground truth).\n"
                "2. KHÔNG được thêm bớt các sự thật bên ngoài không có trong dữ liệu đầu vào hoặc kết quả phân tích.\n"
                "3. KHÔNG được tự ý đưa ra phán quyết khác với kết quả đã cho.\n"
                "4. Chỉ giải thích lý do tại sao kết quả đó xảy ra dựa trên dữ liệu đầu vào và các yếu tố thành công/thách thức được cung cấp.\n"
                "5. Luôn tuân thủ định dạng phản hồi được yêu cầu.\n\n"
                "Ví dụ định dạng khi rủi ro cao (WARNING):\n"
                "\"Xin lỗi, dự án này được đánh giá có rủi ro cao (X% failed). Mặc dù có [yếu tố thành công 1], nhưng [thách thức 1] kết hợp với [thách thức 2] sẽ gây khó khăn lớn cho tiến độ. Tôi đề xuất: (1) Tăng thời gian dự án (2) Bổ sung nhân sự hoặc (3) Tối ưu hóa ngân sách.\"\n\n"
                "Ví dụ định dạng khi cần xem xét kỹ (REVIEW):\n"
                "\"Dự án được đánh giá mức độ an toàn cao nhưng cần lưu ý (X% success/review). Có ưu điểm: [yếu tố thành công 1], tuy nhiên tồn tại thách thức: [thách thức 1] cần chú ý. Tôi đề xuất: (1) [đề xuất 1] (2) [đề xuất 2]\"\n\n"
                "Ví dụ định dạng khi được duyệt (APPROVED):\n"
                "\"Chúc mừng, dự án này được đánh giá an toàn (X% success). Với [yếu tố thành công 1], việc hoàn thành dự án là khả thi. Tôi đề xuất: (1) Tiếp tục theo dõi ngân sách (2) Duy trì tiến độ hiện tại.\""
            )

            user_prompt = (
                f"Thông tin dự án:\n"
                f"- Số ngày thực hiện: {request.Project_Duration_Days} ngày\n"
                f"- Ngân sách dự kiến: {request.Expected_Budget}\n"
                f"- Quy mô đội ngũ: {request.Team_Size} người\n"
                f"- Trình độ trung bình: {request.Avg_Team_Skill_Level}\n"
                f"- Điểm độ phức tạp: {request.Complexity_Score}\n"
                f"- Tỷ lệ sử dụng ngân sách: {request.Budget_Utilization * 100:.1f}%\n\n"
                f"Phân tích ML & Nghiệp vụ:\n"
                f"- ML Prediction: {ml_res['prediction_label']} (Mã: {ml_res['prediction_code']})\n"
                f"- ML Confidence: {ml_res['confidence_score']:.1f}%\n"
                f"- Trạng thái nghiệp vụ: {business_status_text} ({business_status_code})\n"
                f"- Yếu tố thành công: {', '.join(success_factors) if success_factors else 'Không có'}\n"
                f"- Thách thức: {', '.join(potential_challenges) if potential_challenges else 'Không có'}\n\n"
                f"Hãy viết một đoạn phân tích bằng tiếng Việt theo định dạng mẫu tương ứng."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            try:
                # Async explanation call to avoid blocking the thread
                response = await llm.ainvoke(messages)
                llm_insight = str(response.content)
                explanation_source = "llm_explanation"
                
                # Count tokens
                prompt_tokens = 0
                completion_tokens = 0
                if hasattr(response, "response_metadata") and response.response_metadata:
                    meta = response.response_metadata
                    usage = meta.get("token_usage") or meta.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0

                model_name = request.model or LLMFactory._default_models.get((request.provider or settings.LLM_PROVIDER).lower(), "gpt-4o-mini")
                if not prompt_tokens:
                    prompt_str = system_prompt + " " + user_prompt
                    prompt_tokens = await count_tokens(prompt_str, llm=llm, model_name=model_name)
                if not completion_tokens:
                    completion_tokens = await count_tokens(llm_insight, llm=llm, model_name=model_name)

                token_usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            except Exception as exc:
                print(f"[!] LLM execution failed, calling local template fallback: {exc}")
                success_str = ", ".join(success_factors) if success_factors else "sử dụng phương pháp phù hợp"
                challenges_str = ", ".join(potential_challenges) if potential_challenges else "không có cảnh báo nghiêm trọng"
                
                # Local fallback template formatting (dynamic facts check)
                if business_status_code == "WARNING":
                    llm_insight = (
                        f"Xin lỗi, dự án này được đánh giá có rủi ro cao ({ml_res['confidence_score']:.0f}% failed). "
                        f"Mặc dù có ưu điểm: {success_str}, nhưng các thách thức: {challenges_str} sẽ gây khó khăn lớn cho tiến độ. "
                        f"Tôi đề xuất: (1) Tăng thời gian dự án, (2) Bổ sung nhân sự hoặc (3) Tối ưu hóa việc phân bổ ngân sách."
                    )
                elif business_status_code == "REVIEW":
                    llm_insight = (
                        f"Dự án được đánh giá mức độ an toàn cao nhưng cần lưu ý ({ml_res['confidence_score']:.0f}% success/review). "
                        f"Có ưu điểm: {success_str}, tuy nhiên tồn tại thách thức: {challenges_str}. "
                        f"Tôi đề xuất: (1) Giám sát chặt chẽ các rủi ro đã nêu, (2) Thiết lập phương án dự phòng ngân sách/nhân sự."
                    )
                else:
                    llm_insight = (
                        f"Chúc mừng, dự án này được đánh giá an toàn ({ml_res['confidence_score']:.0f}% success). "
                        f"Với thế mạnh: {success_str}, việc hoàn thành dự án đúng hạn là hoàn toàn khả thi. "
                        f"Tôi đề xuất: (1) Tiếp tục duy trì hiệu suất đội ngũ hiện tại, (2) Định kỳ cập nhật tiến độ công việc."
                    )
                explanation_source = "local_explanation_fallback"

        # 5. Return response schema
        return ProjectRiskAssessmentResponse(
            prediction_label=ml_res["prediction_label"],
            prediction_code=ml_res["prediction_code"],
            class_probabilities=ml_res["class_probabilities"],
            confidence_score=ml_res["confidence_score"],
            business_status_code=business_status_code,
            business_status_text=business_status_text,
            success_factors=success_factors,
            potential_challenges=potential_challenges,
            llm_insight=llm_insight,
            explanation_source=explanation_source,
            usage=token_usage
        )

project_risk_assessment_service = ProjectRiskAssessmentService()
