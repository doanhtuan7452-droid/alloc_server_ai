from pydantic import BaseModel

class ProjectData(BaseModel):
    Project_Duration_Days: int
    Expected_Budget: float
    Team_Size: int
    Avg_Team_Skill_Level: float
    Complexity_Score: float
    Budget_Utilization: float
    Methodology_Used_Kanban: int
    Methodology_Used_Scrum: int
    Methodology_Used_Waterfall: int
    Methodology_Used_Hybrid: int
