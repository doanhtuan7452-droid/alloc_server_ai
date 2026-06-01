from fastapi import HTTPException
from langchain_core.messages import SystemMessage, HumanMessage
from app.schemas.assessment import PersonnelAssessmentRequest, AllocationAssessmentResponse
from app.services.employee import employee_service
from app.services.llm_factory import LLMFactory

class AllocationAssessmentService:
    async def assess(self, request: PersonnelAssessmentRequest, bypass_llm: bool = False) -> AllocationAssessmentResponse:
        # 1. Ensure the ML recommendation model is loaded
        employee_service.ensure_loaded()

        # 2. Adapt request
        adapter_res = request.to_allocation_request()

        # 3. Predict ML probabilities
        ml_res = employee_service.predict_with_probabilities(adapter_res.allocation_request)

        # 4. Apply Business Rules Engine
        success_factors = []
        potential_challenges = []

        # Resolve the actual values used during ML model preprocessing
        skill_map = {"low": 0, "medium": 1, "high": 2, "expert": 3}
        skill_level_str = request.skill_level.lower().strip()
        req_skill_level_str = (request.required_skill_level or "medium").lower().strip()
        
        skill_val = skill_map.get(skill_level_str, 1)
        req_skill_val = skill_map.get(req_skill_level_str, 1)
        skill_gap = skill_val - req_skill_val
        
        workload_hours = request.workload_hours if request.workload_hours is not None else 40.0
        hours_per_day = workload_hours / (request.deadline_days + 1e-5)
        
        performance_rating_str = (request.performance_rating or "average").lower().strip()
        conflict_rate = request.conflict_rate if request.conflict_rate is not None else 5.0
        attendance_rate = request.attendance_rate if request.attendance_rate is not None else 95.0

        # Enriched Success Factors classification
        if request.communication_score >= 85:
            success_factors.append("Kỹ năng giao tiếp xuất sắc (score >= 85)")
        if request.technical_skill_score >= 85:
            success_factors.append("Kỹ năng chuyên môn xuất sắc (score >= 85)")
        if request.experience_years >= 5:
            success_factors.append("Kinh nghiệm phong phú (>= 5 năm)")
        if skill_gap >= 1:
            success_factors.append(f"Trình độ vượt trội yêu cầu (skill gap +{skill_gap})")
        if performance_rating_str in ("excellent", "outstanding"):
            success_factors.append(f"Hiệu suất làm việc nổi bật ({performance_rating_str})")

        # Enriched Potential Challenges classification
        if request.experience_years <= 2:
            potential_challenges.append("Kinh nghiệm hạn chế (<= 2 năm)")
        if request.deadline_days <= 7:
            potential_challenges.append("Thời gian hoàn thành quá ngắn (<= 7 ngày)")
        if request.task_complexity in ("high", "critical"):
            potential_challenges.append("Độ phức tạp công việc cao (high/critical)")
        if request.technical_skill_score < 60:
            potential_challenges.append("Kỹ năng chuyên môn yếu (< 60)")
        if request.communication_score < 60:
            potential_challenges.append("Kỹ năng giao tiếp yếu (< 60)")
        if skill_gap < 0:
            potential_challenges.append(f"Thiếu hụt trình độ yêu cầu (skill gap {skill_gap})")
        if hours_per_day > 8.0:
            potential_challenges.append(f"Cường độ làm việc quá tải ({hours_per_day:.1f} giờ/ngày)")
        if attendance_rate < 85.0:
            potential_challenges.append(f"Tỷ lệ chuyên cần thấp ({attendance_rate:.1f}%)")
        if conflict_rate > 20.0:
            potential_challenges.append(f"Tỷ lệ xung đột nhóm cao ({conflict_rate:.1f}%)")
        if performance_rating_str in ("poor", "average"):
            potential_challenges.append(f"Hiệu suất làm việc chưa cao ({performance_rating_str})")
        if adapter_res.confidence_penalty >= 15.0:
            potential_challenges.append("Độ tin cậy dữ liệu đầu vào thấp (thiếu nhiều trường thông tin)")

        # Severity Classification
        # - WARNING: ML predicts failure (code 1 or 2)
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

        # 5. LLM Explanation (or bypass)
        if bypass_llm:
            llm_insight = ""
            explanation_source = "bypassed"
        else:
            # Construct Prompt
            llm = LLMFactory.get_llm(
                provider=request.provider,
                model=request.model,
                temperature=request.temperature
            )

            system_prompt = (
                "Bạn là một trợ lý AI chuyên gia giải thích rủi ro và phân bổ nhân sự.\n"
                "Nhiệm vụ của bạn là giải thích kết quả đánh giá cho người dùng bằng tiếng Việt.\n"
                "RẤT QUAN TRỌNG:\n"
                "1. KHÔNG được thay đổi kết luận của mô hình ML và các luật nghiệp vụ. Kết luận đó là chân lý nền tảng (ground truth).\n"
                "2. KHÔNG được thêm bớt các sự thật bên ngoài không có trong dữ liệu đầu vào hoặc kết quả phân tích.\n"
                "3. KHÔNG được tự ý đưa ra phán quyết khác với kết quả đã cho.\n"
                "4. Chỉ giải thích lý do tại sao kết quả đó xảy ra dựa trên dữ liệu đầu vào và các yếu tố thành công/thách thức được cung cấp.\n"
                "5. Luôn tuân thủ định dạng phản hồi được yêu cầu.\n\n"
                "Ví dụ định dạng khi rủi ro cao (WARNING):\n"
                "\"Xin lỗi, nhân viên này có rủi ro cao (X% failed). Mặc dù có [yếu tố thành công 1], nhưng [thách thức 1] kết hợp với [thách thức 2] sẽ khó khăn. Tôi đề xuất: (1) [đề xuất 1] (2) [đề xuất 2] hoặc (3) [đề xuất 3]\"\n\n"
                "Ví dụ định dạng khi cần xem xét kỹ (REVIEW):\n"
                "\"Kết quả phân bổ cần xem xét kỹ (X% success/review). Nhân viên có điểm mạnh: [yếu tố thành công 1], tuy nhiên có thách thức: [thách thức 1] sẽ cần được lưu ý. Tôi đề xuất: (1) [đề xuất 1] (2) [đề xuất 2]\"\n\n"
                "Ví dụ định dạng khi được duyệt (APPROVED):\n"
                "\"Chúc mừng, nhân viên này được đánh giá an toàn (X% success). Với [yếu tố thành công 1], việc hoàn thành công việc là hoàn toàn khả thi. Tôi đề xuất: (1) [đề xuất 1] (2) [đề xuất 2].\""
            )

            user_prompt = (
                f"Thông tin nhân viên:\n"
                f"- Số năm kinh nghiệm: {request.experience_years} năm\n"
                f"- Điểm kỹ năng chuyên môn: {request.technical_skill_score}/100\n"
                f"- Điểm kỹ năng giao tiếp: {request.communication_score}/100\n"
                f"- Độ phức tạp công việc: {request.task_complexity}\n"
                f"- Hạn chót: {request.deadline_days} ngày\n\n"
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
            except Exception as exc:
                print(f"[!] LLM execution failed, calling local template fallback: {exc}")
                success_str = ", ".join(success_factors) if success_factors else "đầy đủ kỹ năng cơ bản"
                challenges_str = ", ".join(potential_challenges) if potential_challenges else "không có thách thức lớn"
                
                # Local fallback template formatting (dynamic facts check)
                if business_status_code == "WARNING":
                    llm_insight = (
                        f"Xin lỗi, nhân viên này có rủi ro cao ({ml_res['confidence_score']:.0f}% failed). "
                        f"Mặc dù có ưu điểm: {success_str}, nhưng các thách thức: {challenges_str} sẽ khó khăn. "
                        f"Tôi đề xuất: (1) Tìm senior mentor, (2) Xem xét tăng deadline hoặc (3) Giảm độ phức tạp công việc."
                    )
                elif business_status_code == "REVIEW":
                    llm_insight = (
                        f"Kết quả phân bổ cần xem xét kỹ ({ml_res['confidence_score']:.0f}% success/review). "
                        f"Nhân viên có thế mạnh: {success_str}, tuy nhiên có thách thức: {challenges_str}. "
                        f"Tôi đề xuất: (1) Tìm kiếm senior mentor kèm cặp, (2) Thường xuyên theo dõi sát tiến độ."
                    )
                else:
                    llm_insight = (
                        f"Chúc mừng, nhân viên này được đánh giá an toàn ({ml_res['confidence_score']:.0f}% success). "
                        f"Với ưu điểm nổi bật: {success_str}, việc hoàn thành công việc là hoàn toàn khả thi. "
                        f"Tôi đề xuất: (1) Bắt đầu công việc ngay lập tức, (2) Duy trì và cập nhật tiến độ định kỳ."
                    )
                explanation_source = "local_explanation_fallback"

        # 6. Return response schema
        return AllocationAssessmentResponse(
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
            assumptions=adapter_res.assumptions,
            missing_fields=adapter_res.missing_fields,
            confidence_penalty=adapter_res.confidence_penalty
        )

allocation_assessment_service = AllocationAssessmentService()
