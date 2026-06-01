import sys
import os
import unittest
from fastapi.testclient import TestClient

# Ensure sys.path includes the current working directory
sys.path.insert(0, os.getcwd())

from api_server import app
from app.services.employee import employee_service
from app.services.project_risk import project_risk_service
from app.services.allocation_assessment import allocation_assessment_service
from app.services.project_risk_assessment import project_risk_assessment_service

client = TestClient(app)

def safe_print(*args):
    msg = " ".join(str(arg) for arg in args)
    print(msg.encode('ascii', errors='backslashreplace').decode('ascii'))

class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure models are loaded
        employee_service.load_models()
        project_risk_service.load_models()

    def test_root_endpoint(self):
        safe_print("\n>>> Testing Root Endpoint...")
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        safe_print("Root endpoint verified successfully.")

    def test_schema_validations(self):
        safe_print("\n>>> Testing Schema Validations (Invalid inputs)...")
        # 1. Negative experience years
        payload = {
            "experience_years": -1.0,
            "skill_level": "medium",
            "technical_skill_score": 80.0,
            "communication_score": 85.0,
            "task_complexity": "medium",
            "deadline_days": 10
        }
        res = client.post("/api/v1/allocation/assess", json=payload)
        self.assertEqual(res.status_code, 422)
        safe_print("Negative experience years correctly blocked (422).")

        # 2. Score > 100
        payload["experience_years"] = 3.0
        payload["technical_skill_score"] = 105.0
        res = client.post("/api/v1/allocation/assess", json=payload)
        self.assertEqual(res.status_code, 422)
        safe_print("Out-of-bounds technical score correctly blocked (422).")

        # 3. Deadline = 0 (Must be gt=0)
        payload["technical_skill_score"] = 80.0
        payload["deadline_days"] = 0
        res = client.post("/api/v1/allocation/assess", json=payload)
        self.assertEqual(res.status_code, 422)
        safe_print("Zero deadline days correctly blocked (422).")

        # 4. Invalid Literal
        payload["deadline_days"] = 10
        payload["task_complexity"] = "extremely critical"
        res = client.post("/api/v1/allocation/assess", json=payload)
        self.assertEqual(res.status_code, 422)
        safe_print("Invalid literal task_complexity correctly blocked (422).")

    def test_allocation_adapter_and_penalty(self):
        safe_print("\n>>> Testing Allocation Adapter & Confidence Penalty...")
        # 1. No optional fields (Confidence penalty should be 25.0)
        payload = {
            "experience_years": 3.0,
            "skill_level": "medium",
            "technical_skill_score": 80.0,
            "communication_score": 80.0,
            "task_complexity": "medium",
            "deadline_days": 10
        }
        res = client.post("/api/v1/allocation/assess", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["confidence_penalty"], 25.0)
        self.assertEqual(len(data["missing_fields"]), 10)
        safe_print("No-optional-field penalty correctly calculated (25.0).")

        # 2. Provide 4 optional fields (Missing = 6, Penalty = 15.0)
        payload.update({
            "education_level": "master",
            "leadership_score": 70.0,
            "workload_hours": 30.0,
            "team_size": 4
        })
        res = client.post("/api/v1/allocation/assess", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["confidence_penalty"], 15.0)
        self.assertEqual(len(data["missing_fields"]), 6)
        safe_print("Partial optional field penalty correctly calculated (15.0).")

    def test_severity_levels(self):
        safe_print("\n>>> Testing Business Severity Status Levels...")
        # 1. Expecting REVIEW (ml_res Optimal but missing fields / average performance rating)
        payload = {
            "experience_years": 2.5,
            "skill_level": "medium",
            "technical_skill_score": 80.0,
            "communication_score": 85.0,
            "task_complexity": "medium",
            "deadline_days": 10
        }
        res = client.post("/api/v1/allocation/assess", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["business_status_code"], "REVIEW")
        safe_print("REVIEW severity level mapped successfully.")

        # 2. Expecting APPROVED (all optional fields filled, no challenges)
        payload.update({
            "education_level": "bachelor",
            "leadership_score": 50.0,
            "problem_solving_score": 50.0,
            "required_skill_level": "medium",
            "workload_hours": 40.0,
            "task_priority": "medium",
            "team_size": 3,
            "attendance_rate": 95.0,
            "performance_rating": "excellent",
            "conflict_rate": 5.0
        })
        res = client.post("/api/v1/allocation/assess", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["business_status_code"], "APPROVED")
        safe_print("APPROVED severity level mapped successfully.")

    def test_project_risk_normalization(self):
        safe_print("\n>>> Testing Budget_Utilization Normalization...")
        payload = {
            "Project_Duration_Days": 45,
            "Expected_Budget": 150000.0,
            "Team_Size": 4,
            "Avg_Team_Skill_Level": 3.2,
            "Complexity_Score": 3.0,
            "Budget_Utilization": 85.0, # percentage
            "Methodology_Used_Kanban": 0,
            "Methodology_Used_Scrum": 1,
            "Methodology_Used_Waterfall": 0,
            "Methodology_Used_Hybrid": 0
        }
        res = client.post("/api/v1/project-risk/assess", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("class_probabilities", data)
        safe_print("Budget_Utilization successfully normalized and assessed.")

    def test_service_unavailability_error(self):
        safe_print("\n>>> Testing Model Unavailability (HTTP 503)...")
        # Temporarily nullify model
        old_model = employee_service.model
        employee_service.model = None
        try:
            payload = {
                "experience_years": 3.0,
                "skill_level": "medium",
                "technical_skill_score": 80.0,
                "communication_score": 80.0,
                "task_complexity": "medium",
                "deadline_days": 10
            }
            res = client.post("/api/v1/allocation/assess", json=payload)
            self.assertEqual(res.status_code, 503)
            safe_print("Service Unavailability properly returns 503.")
        finally:
            # Restore model
            employee_service.model = old_model

if __name__ == "__main__":
    unittest.main()
