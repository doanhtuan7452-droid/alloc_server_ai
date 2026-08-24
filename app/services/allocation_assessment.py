from typing import Union, List, Any
from fastapi import HTTPException
from langchain_core.messages import SystemMessage, HumanMessage
from app.schemas.assessment import (
    PersonnelAssessmentRequest,
    BulkAssessmentRequest,
    EmployeeAssessmentInput,
    AllocationAssessmentResponse,
    EmployeeAllocationAssessmentResult,
    BulkAllocationAssessmentResponse
)
from app.schemas.mongo_chat import TokenUsage
from fastapi.concurrency import run_in_threadpool
from app.services.employee import employee_service
from app.services.fit_regressor import fit_regressor_service
from app.services.llm_factory import LLMFactory
from app.services.memory_service import count_tokens
from app.services.skill_filter_service import skill_filter_service, CandidateSkillFilterResult
from app.core.config import settings
import asyncio

class AllocationAssessmentService:
    async def assess(
        self,
        request: Union[PersonnelAssessmentRequest, BulkAssessmentRequest],
        bypass_llm: bool = False,
        db: Any = None
    ) -> Union[AllocationAssessmentResponse, BulkAllocationAssessmentResponse]:
        # 1. Ensure the ML recommendation model and Fit Regressor model are loaded
        employee_service.ensure_loaded()
        fit_regressor_service.ensure_loaded()

        # 2. Check request type
        is_bulk = request.request_type == "bulk"

        # 3. Prepare task-level fields
        task_name = getattr(request, "task_name", None)
        task_complexity = request.task_complexity
        deadline_days = request.deadline_days
        required_skill_level = request.required_skill_level
        workload_hours = request.workload_hours
        task_priority = request.task_priority
        team_size = request.team_size

        # 4. Prepare candidates list
        if is_bulk:
            candidates = request.employees
        else:
            # Create a single temporary candidate from root fields
            candidates = [
                EmployeeAssessmentInput(
                    employee_id="EMP-SINGLE",
                    employee_name="Single Employee",
                    experience_years=request.experience_years,
                    skill_level=request.skill_level,
                    technical_skill_score=request.technical_skill_score,
                    communication_score=request.communication_score,
                    education_level=request.education_level,
                    leadership_score=request.leadership_score,
                    problem_solving_score=request.problem_solving_score,
                    attendance_rate=request.attendance_rate,
                    performance_rating=request.performance_rating,
                    conflict_rate=request.conflict_rate,
                    skills=getattr(request, "skills", []) or []
                )
            ]

        # 5. NLP / Embedding Skill Filtering: Lọc kỹ năng & tính EffectiveTechScore
        skill_filter_results = await skill_filter_service.filter_bulk_candidate_skills(
            task_name=task_name,
            candidates=candidates,
            provider=request.provider or settings.LLM_PROVIDER,
            model=request.model,
            db=db
        )

        # Map each candidate to AllocationRequest with effective technical skill score
        allocation_requests = []
        adapter_results = []
        for candidate, filter_res in zip(candidates, skill_filter_results):
            # Tạo bản sao candidate đã được hiệu chỉnh điểm kỹ năng thực tế
            adjusted_candidate = candidate.model_copy(update={
                "technical_skill_score": filter_res.effective_tech_score,
                "skill_level": filter_res.derived_skill_level
            })

            adapter_res = adjusted_candidate.to_allocation_request(
                task_complexity=task_complexity,
                deadline_days=deadline_days,
                required_skill_level=required_skill_level,
                workload_hours=workload_hours,
                task_priority=task_priority,
                team_size=team_size
            )
            allocation_requests.append(adapter_res.allocation_request)
            adapter_results.append(adapter_res)

        # 6. Run ML predictions vectorized in a single batch
        ml_results = employee_service.predict_batch_with_probabilities(allocation_requests)

        # 6b. Run Fit Regressor model predictions vectorized in a single batch via Worker Thread Pool
        fit_percentages = await run_in_threadpool(
            fit_regressor_service.predict_batch,
            allocation_requests
        )

        # 7. Setup LLM concurrency semaphore (max 5 concurrent calls)
        semaphore = asyncio.Semaphore(5)

        async def _assess_single_candidate_with_llm(
            candidate: EmployeeAssessmentInput,
            filter_res: CandidateSkillFilterResult,
            adapter_res,
            ml_res: dict,
            fit_percentage: float
        ) -> EmployeeAllocationAssessmentResult:
            try:
                # Apply Business Rules Engine
                success_factors = []
                potential_challenges = []

                # Resolve the actual values used during ML model preprocessing
                skill_map = {"low": 0, "medium": 1, "high": 2, "expert": 3}
                skill_level_str = filter_res.derived_skill_level.lower().strip()
                req_skill_level_str = (required_skill_level or "medium").lower().strip()
                
                skill_val = skill_map.get(skill_level_str, 1)
                req_skill_val = skill_map.get(req_skill_level_str, 1)
                skill_gap = skill_val - req_skill_val
                
                actual_workload_hours = workload_hours if workload_hours is not None else 40.0
                hours_per_day = actual_workload_hours / (deadline_days + 1e-5)
                
                performance_rating_str = (candidate.performance_rating or "average").lower().strip()
                conflict_rate = candidate.conflict_rate if candidate.conflict_rate is not None else 5.0
                attendance_rate = candidate.attendance_rate if candidate.attendance_rate is not None else 95.0

                # Enriched Success Factors classification
                if candidate.communication_score >= 85:
                    success_factors.append("Kỹ năng giao tiếp xuất sắc (score >= 85)")
                if filter_res.effective_tech_score >= 80:
                    success_factors.append("Kỹ năng chuyên môn thực tế xuất sắc (score >= 80)")
                if filter_res.matched_skills and filter_res.semantic_skill_score >= 60:
                    matched_names = ", ".join(s.skill_name for s in filter_res.matched_skills)
                    success_factors.append(f"Kỹ năng khớp tốt với yêu cầu Task ({matched_names})")
                if candidate.experience_years >= 5:
                    success_factors.append("Kinh nghiệm phong phú (>= 5 năm)")
                if skill_gap >= 1:
                    success_factors.append(f"Trình độ vượt trội yêu cầu (skill gap +{skill_gap})")
                if performance_rating_str in ("excellent", "outstanding"):
                    success_factors.append(f"Hiệu suất làm việc nổi bật ({performance_rating_str})")

                # Enriched Potential Challenges classification
                if candidate.experience_years <= 2:
                    potential_challenges.append("Kinh nghiệm hạn chế (<= 2 năm)")
                if deadline_days <= 7:
                    potential_challenges.append("Thời gian hoàn thành quá ngắn (<= 7 ngày)")
                if task_complexity in ("high", "critical"):
                    potential_challenges.append("Độ phức tạp công việc cao (high/critical)")
                if filter_res.effective_tech_score < 60:
                    potential_challenges.append("Kỹ năng chuyên môn thực tế chưa đạt mức kỳ vọng (< 60)")
                if not filter_res.matched_skills and task_name:
                    potential_challenges.append("Không có kỹ năng chuyên môn phù hợp với yêu cầu của Task")
                elif filter_res.is_marginal_match:
                    marginal_names = ", ".join(s.skill_name for s in filter_res.matched_skills)
                    potential_challenges.append(f"Kỹ năng chỉ liên quan gián tiếp đến Task ({marginal_names})")
                if candidate.communication_score < 60:
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
                if ml_res["prediction_code"] != 0:
                    business_status_code = "WARNING"
                    business_status_text = "CẢNH BÁO"
                elif potential_challenges:
                    business_status_code = "REVIEW"
                    business_status_text = "CẦN XEM XÉT"
                else:
                    business_status_code = "APPROVED"
                    business_status_text = "CHẤP THUẬN"

                # Chuẩn bị thông tin kỹ năng đã lọc cho Prompt
                if filter_res.matched_skills:
                    matched_str = ", ".join(f"{s.skill_name} (Level {s.level})" for s in filter_res.matched_skills)
                    skill_matching_info = f"- Kỹ năng chuyên môn khớp với Task: {matched_str} (Điểm tương quan ngữ nghĩa: {filter_res.semantic_skill_score}/100)"
                    if filter_res.is_marginal_match:
                        skill_matching_info += " [Lưu ý: Kỹ năng chỉ có mức độ liên quan gián tiếp]"
                else:
                    skill_matching_info = "- Kỹ năng chuyên môn khớp với Task: Không có kỹ năng nào phù hợp (Điểm tương quan ngữ nghĩa: 0/100)"

                # LLM Explanation (or bypass)
                token_usage = TokenUsage()
                if bypass_llm:
                    llm_insight = ""
                    explanation_source = "bypassed"
                else:
                    async def call_llm_with_semaphore():
                        async with semaphore:
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
                                "4. Chỉ giải thích lý do tại sao kết quả đó xảy ra dựa trên dữ liệu đầu vào, các kỹ năng khớp và các yếu tố thành công/thách thức được cung cấp.\n"
                                "5. Luôn tuân thủ định dạng phản hồi được yêu cầu.\n\n"
                                "Ví dụ định dạng khi rủi ro cao (WARNING):\n"
                                "\"Xin lỗi, nhân viên này có rủi ro cao (X% failed). Mặc dù có [yếu tố thành công 1], nhưng [thách thức 1] kết hợp với [thách thức 2] sẽ khó khăn. Tôi đề xuất: (1) [đề xuất 1] (2) [đề xuất 2] hoặc (3) [đề xuất 3]\"\n\n"
                                "Ví dụ định dạng khi cần xem xét kỹ (REVIEW):\n"
                                "\"Kết quả phân bổ cần xem xét kỹ (X% success/review). Nhân viên có điểm mạnh: [yếu tố thành công 1], tuy nhiên có thách thức: [thách thức 1] sẽ cần được lưu ý. Tôi đề xuất: (1) [đề xuất 1] (2) [đề xuất 2]\"\n\n"
                                "Ví dụ định dạng khi được duyệt (APPROVED):\n"
                                "\"Chúc mừng, nhân viên này được đánh giá an toàn (X% success). Với [yếu tố thành công 1], việc hoàn thành công việc là hoàn toàn khả thi. Tôi đề xuất: (1) [đề xuất 1] (2) [đề xuất 2].\""
                            )

                            task_info_str = f"Tên công việc (Task): {task_name}\n" if task_name else ""
                            user_prompt = (
                                f"Thông tin nhiệm vụ & Nhân viên:\n"
                                f"{task_info_str}"
                                f"- Số năm kinh nghiệm: {candidate.experience_years} năm\n"
                                f"- Điểm kỹ năng chuyên môn thực tế (đã hiệu chỉnh theo task): {filter_res.effective_tech_score}/100 (Điểm HR gốc: {candidate.technical_skill_score}/100)\n"
                                f"- Điểm kỹ năng giao tiếp: {candidate.communication_score}/100\n"
                                f"{skill_matching_info}\n"
                                f"- Độ phức tạp công việc: {task_complexity}\n"
                                f"- Hạn chót: {deadline_days} ngày\n\n"
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

                            response = await llm.ainvoke(messages)
                            
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
                                completion_tokens = await count_tokens(str(response.content), llm=llm, model_name=model_name)

                            usage_obj = TokenUsage(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=prompt_tokens + completion_tokens
                            )
                            return str(response.content), "llm_explanation", usage_obj

                    try:
                        llm_insight, explanation_source, token_usage = await call_llm_with_semaphore()
                    except Exception as exc:
                        print(f"[!] LLM execution failed for employee {candidate.employee_id}, calling local template fallback: {exc}")
                        success_str = ", ".join(success_factors) if success_factors else "đầy đủ kỹ năng cơ bản"
                        challenges_str = ", ".join(potential_challenges) if potential_challenges else "không có thách thức lớn"
                        
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

                return EmployeeAllocationAssessmentResult(
                    employee_id=candidate.employee_id,
                    employee_name=candidate.employee_name,
                    fit_percentage=round(fit_percentage, 1),
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
                    confidence_penalty=adapter_res.confidence_penalty,
                    matched_skills=filter_res.matched_skills,
                    semantic_skill_score=filter_res.semantic_skill_score,
                    is_marginal_match=filter_res.is_marginal_match,
                    usage=token_usage
                )
            except Exception as candidate_err:
                print(f"[ERROR] Candidate {candidate.employee_id} assessment error: {candidate_err}")
                return EmployeeAllocationAssessmentResult(
                    employee_id=candidate.employee_id,
                    employee_name=candidate.employee_name,
                    fit_percentage=0.0,
                    prediction_label=ml_res.get("prediction_label", "Suboptimal"),
                    prediction_code=ml_res.get("prediction_code", 1),
                    class_probabilities=ml_res.get("class_probabilities", {}),
                    confidence_score=ml_res.get("confidence_score", 0.0),
                    business_status_code="WARNING",
                    business_status_text="LỖI ĐÁNH GIÁ",
                    success_factors=[],
                    potential_challenges=["Lỗi hệ thống khi phân tích ứng viên"],
                    llm_insight=f"Đánh giá ứng viên thất bại do lỗi hệ thống: {str(candidate_err)}",
                    explanation_source="system_error",
                    assumptions=adapter_res.assumptions if 'adapter_res' in locals() else {},
                    missing_fields=adapter_res.missing_fields if 'adapter_res' in locals() else [],
                    confidence_penalty=adapter_res.confidence_penalty if 'adapter_res' in locals() else 0.0,
                    matched_skills=filter_res.matched_skills if 'filter_res' in locals() else [],
                    semantic_skill_score=filter_res.semantic_skill_score if 'filter_res' in locals() else 0.0,
                    is_marginal_match=filter_res.is_marginal_match if 'filter_res' in locals() else False,
                    usage=TokenUsage()
                )

        # 8. Collect results asynchronously using gather
        tasks = [
            _assess_single_candidate_with_llm(candidate, filter_res, adapter_res, ml_res, fit_pct)
            for candidate, filter_res, adapter_res, ml_res, fit_pct in zip(
                candidates, skill_filter_results, adapter_results, ml_results, fit_percentages
            )
        ]

        # Use return_exceptions=True to keep other tasks running even if one raises a raw exception
        gathered_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Resolve gathered_results to handle any raw Exception objects
        resolved_results = []
        for i, res in enumerate(gathered_results):
            if isinstance(res, Exception):
                print(f"[CRITICAL] Exception returned from candidate task {candidates[i].employee_id}: {res}")
                resolved_results.append(
                    EmployeeAllocationAssessmentResult(
                        employee_id=candidates[i].employee_id,
                        employee_name=candidates[i].employee_name,
                        fit_percentage=0.0,
                        prediction_label="Suboptimal",
                        prediction_code=1,
                        class_probabilities={},
                        confidence_score=0.0,
                        business_status_code="WARNING",
                        business_status_text="LỖI ĐÁNH GIÁ",
                        success_factors=[],
                        potential_challenges=["Lỗi hệ thống nghiêm trọng"],
                        llm_insight=f"Đã xảy ra ngoại lệ bất ngờ: {str(res)}",
                        explanation_source="system_exception",
                        assumptions={},
                        missing_fields=[],
                        confidence_penalty=0.0,
                        matched_skills=[],
                        semantic_skill_score=0.0,
                        is_marginal_match=False,
                        usage=TokenUsage()
                    )
                )
            else:
                resolved_results.append(res)

        # 9. Return response based on request mode
        if is_bulk:
            total_prompt = sum(r.usage.prompt_tokens for r in resolved_results)
            total_completion = sum(r.usage.completion_tokens for r in resolved_results)
            total_tokens = sum(r.usage.total_tokens for r in resolved_results)
            bulk_usage = TokenUsage(
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
                total_tokens=total_tokens
            )
            return BulkAllocationAssessmentResponse(results=resolved_results, usage=bulk_usage)
        else:
            single_res = resolved_results[0]
            return AllocationAssessmentResponse(
                prediction_label=single_res.prediction_label,
                prediction_code=single_res.prediction_code,
                class_probabilities=single_res.class_probabilities,
                confidence_score=single_res.confidence_score,
                business_status_code=single_res.business_status_code,
                business_status_text=single_res.business_status_text,
                success_factors=single_res.success_factors,
                potential_challenges=single_res.potential_challenges,
                llm_insight=single_res.llm_insight,
                explanation_source=single_res.explanation_source,
                assumptions=single_res.assumptions,
                missing_fields=single_res.missing_fields,
                confidence_penalty=single_res.confidence_penalty,
                fit_percentage=single_res.fit_percentage,
                matched_skills=single_res.matched_skills,
                semantic_skill_score=single_res.semantic_skill_score,
                is_marginal_match=single_res.is_marginal_match,
                usage=single_res.usage
            )

allocation_assessment_service = AllocationAssessmentService()

