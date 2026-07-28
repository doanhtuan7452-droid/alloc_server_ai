from typing import Literal, Optional
from pydantic import BaseModel

class ProjectData(BaseModel):
    Project_Duration_Days: int
    Expected_Budget: float
    Team_Size: int
    Avg_Team_Skill_Level: float
    Complexity_Score: float
    Budget_Utilization: float
    methodology: Optional[Literal["Agile", "Waterfall", "Scrum", "Kanban", "Hybrid"]] = None
    
    # Binary fields from C# payload
    Methodology_Used_Kanban: Optional[int] = 0
    Methodology_Used_Scrum: Optional[int] = 0
    Methodology_Used_Waterfall: Optional[int] = 0
    Methodology_Used_Hybrid: Optional[int] = 0

