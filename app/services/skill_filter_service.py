import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from app.core.config import settings
from app.schemas.assessment import SkillItemDto
from app.services.tool_rag import tool_rag
from app.services.llm_factory import LLMFactory

logger = logging.getLogger("app.services.skill_filter")

class CandidateSkillFilterResult:
    def __init__(
        self,
        matched_skills: List[SkillItemDto],
        semantic_skill_score: float,
        effective_tech_score: float,
        derived_skill_level: str,
        is_marginal_match: bool = False
    ):
        self.matched_skills = matched_skills
        self.semantic_skill_score = semantic_skill_score
        self.effective_tech_score = effective_tech_score
        self.derived_skill_level = derived_skill_level
        self.is_marginal_match = is_marginal_match

class SkillFilterService:
    def __init__(self):
        # L1 RAM Cache: { cache_key: { normalized_skill_name: list_of_floats } }
        self._ram_cache: Dict[str, Dict[str, List[float]]] = {}

    def _get_cache_key(self, provider: str, model: Optional[str] = None) -> Tuple[str, str]:
        prov = (provider or settings.LLM_PROVIDER).lower().strip()
        mod = model or ("text-embedding-3-small" if prov == "openai" else ("gemini-embedding-001" if prov == "gemini" else settings.OLLAMA_EMBEDDING_MODEL))
        return prov, mod

    async def get_or_create_skills_embeddings_batch(
        self,
        skill_names: List[str],
        provider: str,
        model: Optional[str] = None,
        db: Any = None
    ) -> Dict[str, List[float]]:
        """
        Deduplicate, tra cứu L1 RAM Cache, MongoDB L2 Cache và chỉ sinh embedding hàng loạt
        cho những kỹ năng chưa từng được lưu trong hệ thống.
        """
        if not skill_names:
            return {}

        prov, mod = self._get_cache_key(provider, model)
        cache_key = f"{prov}:{mod}"

        if cache_key not in self._ram_cache:
            self._ram_cache[cache_key] = {}

        # 1. Deduplication: Tập hợp các tên kỹ năng duy nhất đã chuẩn hóa
        unique_skills_map = {s.strip().lower(): s.strip() for s in skill_names if s and s.strip()}
        result_embeddings: Dict[str, List[float]] = {}
        missing_skills_lower: List[str] = []

        # 2. Check L1 RAM Cache
        for s_norm, s_orig in unique_skills_map.items():
            if s_norm in self._ram_cache[cache_key]:
                result_embeddings[s_norm] = self._ram_cache[cache_key][s_norm]
            else:
                missing_skills_lower.append(s_norm)

        # 3. Check MongoDB L2 Cache (nếu có kết nối DB)
        if missing_skills_lower and db is not None:
            try:
                cursor = db["static_skill_embeddings"].find({
                    "skill_name": {"$in": missing_skills_lower},
                    "provider": prov,
                    "model": mod
                })
                async for doc in cursor:
                    s_norm = doc.get("skill_name")
                    emb = doc.get("embedding")
                    if s_norm and emb:
                        emb_floats = [float(x) for x in emb]
                        result_embeddings[s_norm] = emb_floats
                        self._ram_cache[cache_key][s_norm] = emb_floats

                missing_skills_lower = [s for s in missing_skills_lower if s not in result_embeddings]
            except Exception as db_err:
                logger.warning(f"Lỗi khi đọc static_skill_embeddings từ MongoDB: {db_err}")

        # 4. Sinh vector embedding hàng loạt cho các kỹ năng còn thiếu
        if missing_skills_lower:
            texts_to_embed = [unique_skills_map[s] for s in missing_skills_lower]
            logger.info(f"Sinh batch embedding mới cho {len(texts_to_embed)} kỹ năng (Provider: {prov}, Model: {mod}).")
            
            try:
                new_embeddings = await tool_rag.get_embeddings(texts_to_embed, provider=prov)
                if len(new_embeddings) == len(texts_to_embed):
                    mongo_docs_to_insert = []
                    now_utc = datetime.now(timezone.utc)

                    for s_norm, emb in zip(missing_skills_lower, new_embeddings):
                        if emb:
                            emb_floats = [float(x) for x in emb]
                            result_embeddings[s_norm] = emb_floats
                            self._ram_cache[cache_key][s_norm] = emb_floats

                            if db is not None:
                                mongo_docs_to_insert.append({
                                    "skill_name": s_norm,
                                    "original_name": unique_skills_map[s_norm],
                                    "provider": prov,
                                    "model": mod,
                                    "embedding": emb_floats,
                                    "updated_at": now_utc
                                })

                    if mongo_docs_to_insert and db is not None:
                        try:
                            # Upsert từng doc hoặc bulk write để tránh trùng lặp
                            for doc in mongo_docs_to_insert:
                                await db["static_skill_embeddings"].update_one(
                                    {"skill_name": doc["skill_name"], "provider": prov, "model": mod},
                                    {"$set": doc},
                                    upsert=True
                                )
                        except Exception as write_err:
                            logger.error(f"Lỗi khi lưu embeddings vào MongoDB L2 Cache: {write_err}")
                else:
                    logger.warning("Độ dài danh sách vector sinh ra không khớp với danh sách kỹ năng cần embed.")
            except Exception as emb_err:
                logger.error(f"Lỗi trong quá trình sinh batch embedding kỹ năng: {emb_err}")

        return result_embeddings

    async def filter_candidate_skills(
        self,
        task_name: Optional[str],
        skills: List[SkillItemDto],
        technical_skill_score: float,
        provider: str = "openai",
        model: Optional[str] = None,
        db: Any = None
    ) -> CandidateSkillFilterResult:
        """
        Lọc danh sách kỹ năng cho 1 ứng viên đối với task cụ thể, tính toán điểm kỹ năng
        ngữ nghĩa và hiệu chỉnh EffectiveTechScore.
        """
        # Nếu task_name hoặc danh sách kỹ năng rỗng -> Fallback an toàn (không lọc)
        if not task_name or not task_name.strip() or not skills:
            return CandidateSkillFilterResult(
                matched_skills=[],
                semantic_skill_score=0.0,
                effective_tech_score=technical_skill_score,
                derived_skill_level=self._derive_skill_level(technical_skill_score),
                is_marginal_match=False
            )

        prov, mod = self._get_cache_key(provider, model)
        cache_key = f"{prov}:{mod}"

        # 1. Lấy vector embedding của TaskName
        task_clean = task_name.strip()
        task_vector = None
        
        # Check task vector in L1 cache
        task_cache_key = f"__task__:{task_clean.lower()}"
        if cache_key in self._ram_cache and task_cache_key in self._ram_cache[cache_key]:
            task_vector = self._ram_cache[cache_key][task_cache_key]
        else:
            try:
                task_vector = await tool_rag.get_embedding(task_clean, provider=prov)
                if task_vector:
                    if cache_key not in self._ram_cache:
                        self._ram_cache[cache_key] = {}
                    self._ram_cache[cache_key][task_cache_key] = task_vector
            except Exception as e:
                logger.error(f"Lỗi khi sinh embedding cho Task '{task_clean}': {e}")

        if not task_vector:
            # Fallback nếu không embed được task: giữ nguyên điểm số cũ
            return CandidateSkillFilterResult(
                matched_skills=[],
                semantic_skill_score=0.0,
                effective_tech_score=technical_skill_score,
                derived_skill_level=self._derive_skill_level(technical_skill_score),
                is_marginal_match=False
            )

        # 2. Lấy vector embedding của các kỹ năng của ứng viên
        skill_names = [sk.skill_name for sk in skills if sk.skill_name and sk.skill_name.strip()]
        skill_embeddings_map = await self.get_or_create_skills_embeddings_batch(
            skill_names=skill_names,
            provider=prov,
            model=mod,
            db=db
        )

        # 3. Tính toán Cosine Similarity cho từng kỹ năng
        scored_skills: List[Tuple[SkillItemDto, float]] = []
        for sk in skills:
            sk_norm = sk.skill_name.strip().lower()
            sk_vector = skill_embeddings_map.get(sk_norm)
            if sk_vector and len(sk_vector) == len(task_vector):
                sim = tool_rag.cosine_similarity(task_vector, sk_vector)
                scored_skills.append((sk, sim))
            else:
                scored_skills.append((sk, 0.0))

        # 4. Cơ chế lọc lai (Hybrid Matching)
        # Ngưỡng cứng: S_i >= 0.40
        hard_matched = [item for item in scored_skills if item[1] >= 0.40]
        hard_matched.sort(key=lambda x: x[1], reverse=True)

        matched_skills: List[SkillItemDto] = []
        is_marginal_match = False

        if hard_matched:
            # Lấy tối đa Top-5 kỹ năng có điểm tương đồng cao nhất
            matched_items = hard_matched[:5]
            matched_skills = [item[0] for item in matched_items]
            
            # Tính SemanticSkillScore theo công thức chuẩn hóa
            weighted_sum = sum(sim * sk.level for sk, sim in matched_items)
            max_possible_sum = len(matched_items) * 5.0
            semantic_score = (weighted_sum / max_possible_sum) * 100.0 if max_possible_sum > 0 else 0.0
            semantic_score = min(100.0, max(0.0, semantic_score))
        else:
            # Ngoại lệ an toàn: Kiểm tra khoảng tương quan gián tiếp [0.30, 0.40)
            marginal_matched = [item for item in scored_skills if 0.30 <= item[1] < 0.40]
            marginal_matched.sort(key=lambda x: x[1], reverse=True)

            if marginal_matched:
                best_marginal = marginal_matched[0]
                matched_skills = [best_marginal[0]]
                is_marginal_match = True

                weighted_sum = best_marginal[1] * best_marginal[0].level
                max_possible_sum = 1.0 * 5.0
                semantic_score = (weighted_sum / max_possible_sum) * 100.0
                semantic_score = min(100.0, max(0.0, semantic_score))
            else:
                # Ứng viên không có kỹ năng nào đạt ngưỡng tối thiểu
                matched_skills = []
                semantic_score = 0.0
                is_marginal_match = False

        # 5. Tính EffectiveTechScore = 0.4 * HRScore + 0.6 * SemanticScore
        effective_tech_score = (0.4 * technical_skill_score) + (0.6 * semantic_score)
        effective_tech_score = round(min(100.0, max(0.0, effective_tech_score)), 1)
        semantic_score = round(semantic_score, 1)

        derived_skill_level = self._derive_skill_level(effective_tech_score)

        return CandidateSkillFilterResult(
            matched_skills=matched_skills,
            semantic_skill_score=semantic_score,
            effective_tech_score=effective_tech_score,
            derived_skill_level=derived_skill_level,
            is_marginal_match=is_marginal_match
        )

    async def filter_bulk_candidate_skills(
        self,
        task_name: Optional[str],
        candidates: List[Any],
        provider: str = "openai",
        model: Optional[str] = None,
        db: Any = None
    ) -> List[CandidateSkillFilterResult]:
        """
        Xử lý lọc kỹ năng hàng loạt cho toàn bộ danh sách ứng viên trong 1 lượt Bulk Assessment.
        """
        if not candidates:
            return []

        # 1. Gom nhóm deduplicate toàn bộ tên kỹ năng của tất cả ứng viên để batch embedding 1 lần
        all_skill_names = []
        for cand in candidates:
            cand_skills = getattr(cand, "skills", []) or []
            for sk in cand_skills:
                name = getattr(sk, "skill_name", None) or (sk.get("skill_name") if isinstance(sk, dict) else "")
                if name and name.strip():
                    all_skill_names.append(name.strip())

        # Pre-fetch & cache batch embeddings cho toàn bộ kỹ năng
        if all_skill_names:
            await self.get_or_create_skills_embeddings_batch(
                skill_names=all_skill_names,
                provider=provider,
                model=model,
                db=db
            )

        # 2. Xử lý lọc song song cho từng ứng viên
        results = []
        for cand in candidates:
            cand_skills_raw = getattr(cand, "skills", []) or []
            cand_skills = []
            for sk in cand_skills_raw:
                if isinstance(sk, SkillItemDto):
                    cand_skills.append(sk)
                elif isinstance(sk, dict):
                    cand_skills.append(SkillItemDto(
                        skill_name=sk.get("skill_name", ""),
                        level=sk.get("level", 3)
                    ))

            tech_score = float(getattr(cand, "technical_skill_score", 50.0))

            res = await self.filter_candidate_skills(
                task_name=task_name,
                skills=cand_skills,
                technical_skill_score=tech_score,
                provider=provider,
                model=model,
                db=db
            )
            results.append(res)

        return results

    def _derive_skill_level(self, score: float) -> str:
        if score < 40.0:
            return "low"
        elif score >= 75.0:
            return "high"
        return "medium"

skill_filter_service = SkillFilterService()
