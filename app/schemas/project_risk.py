from typing import Literal
from pydantic import BaseModel

class ProjectData(BaseModel):
    Project_Duration_Days: int
    Expected_Budget: float
    Team_Size: int
    Avg_Team_Skill_Level: float
    Complexity_Score: float
    Budget_Utilization: float
    methodology: Literal["Agile", "Waterfall", "Scrum", "Kanban", "Hybrid"]

