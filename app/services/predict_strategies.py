from abc import ABC, abstractmethod
from typing import Dict, Any
from fastapi.concurrency import run_in_threadpool
from app.services.employee import employee_service
from app.services.project_risk import project_risk_service
from app.schemas.allocation import AllocationRequest
from app.schemas.project_risk import ProjectData

class BasePredictStrategy(ABC):
    @abstractmethod
    async def predict(self, data: Any) -> Dict[str, Any]:
        """
        Thực hiện dự báo (CPU-bound) bất đồng bộ thông qua threadpool.
        """
        pass

class AllocationPredictStrategy(BasePredictStrategy):
    async def predict(self, data: AllocationRequest) -> Dict[str, Any]:
        # Offload logic CPU-bound sang threadpool của Starlette
        return await run_in_threadpool(employee_service.predict, data)

class ProjectRiskPredictStrategy(BasePredictStrategy):
    async def predict(self, data: ProjectData) -> Dict[str, Any]:
        # Offload logic CPU-bound sang threadpool của Starlette
        return await run_in_threadpool(project_risk_service.predict, data)

class PredictStrategyFactory:
    _strategies = {
        "allocation": AllocationPredictStrategy(),
        "project_risk": ProjectRiskPredictStrategy()
    }

    @classmethod
    def get_strategy(cls, task_type: str) -> BasePredictStrategy:
        strategy = cls._strategies.get(task_type.lower())
        if not strategy:
            raise ValueError(f"Không hỗ trợ chiến lược dự báo: {task_type}")
        return strategy
