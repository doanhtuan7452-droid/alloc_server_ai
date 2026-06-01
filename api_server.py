from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# ==============================================================================
# 1. KHỞI TẠO ỨNG DỤNG & LOAD MODEL
# ==============================================================================
app = FastAPI(
    title="HR Allocation AI API",
    description="API gợi ý phân bổ nhân sự cho hệ thống dự án C#",
    version="1.0"
)

# Đường dẫn tới thư mục lưu model
EXPORT_DIR = "model_exports"
MODEL_PATH = os.path.join(EXPORT_DIR, "hr_allocation_ai_model.pkl")
SCALER_PATH = os.path.join(EXPORT_DIR, "hr_scaler.pkl")
LE_PATH = os.path.join(EXPORT_DIR, "hr_label_encoder.pkl")

# Load model, scaler và label encoder vào bộ nhớ RAM khi khởi động Server
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LE_PATH)
    print("[*] Đã tải thành công AI Model, Scaler và Label Encoder!")
except Exception as e:
    print(f"[!] Lỗi khi tải Model: {e}")
    print("[!] Hãy chắc chắn thư mục 'model_exports' nằm cùng cấp với file api_server.py")

# ==============================================================================
# 2. ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU ĐẦU VÀO (TỪ C# GỬI SANG)
# ==============================================================================
class AllocationRequest(BaseModel):
    experience_years: float
    education_level: str       # Vd: "Bachelor", "Master"
    skill_level: str           # Vd: "Medium", "High", "Expert"
    technical_skill_score: float
    communication_score: float
    leadership_score: float
    problem_solving_score: float
    task_complexity: str       # Vd: "Low", "Medium", "High"
    required_skill_level: str  # Vd: "Medium", "High"
    deadline_days: int
    workload_hours: float
    task_priority: str         # Vd: "Medium", "Critical"
    team_size: int
    attendance_rate: float
    performance_rating: str    # Vd: "Average", "Excellent"
    conflict_rate: float

# ==============================================================================
# 3. API ENDPOINT ĐỂ DỰ ĐOÁN
# ==============================================================================
@app.post("/api/v1/suggest-allocation")
async def suggest_allocation(request: AllocationRequest):
    try:
        # 1. Chuyển đổi dữ liệu JSON từ C# thành Pandas DataFrame
        input_dict = request.dict()
        df = pd.DataFrame([input_dict])

        # 2. MÃ HÓA (Đồng nhất chữ thành số như lúc Train)
        edu_map = {'high school': 0, 'diploma': 1, 'bachelor': 2, 'master': 3, 'phd': 4}
        perf_map = {'poor': 0, 'average': 1, 'excellent': 2, 'outstanding': 3}
        complex_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        req_skill_map = {'low': 0, 'medium': 1, 'high': 2, 'expert': 3}
        prio_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        skill_map = {'low': 0, 'medium': 1, 'high': 2, 'expert': 3} 

        df['education_level'] = df['education_level'].str.lower().str.strip().map(edu_map).fillna(2)
        df['performance_rating'] = df['performance_rating'].str.lower().str.strip().map(perf_map).fillna(1)
        df['task_complexity'] = df['task_complexity'].str.lower().str.strip().map(complex_map).fillna(1)
        df['required_skill_level'] = df['required_skill_level'].str.lower().str.strip().map(req_skill_map).fillna(1)
        df['task_priority'] = df['task_priority'].str.lower().str.strip().map(prio_map).fillna(1)
        df['skill_level'] = df['skill_level'].str.lower().str.strip().map(skill_map).fillna(1)

        # Cắt mức điểm (0-100)
        score_columns = ['technical_skill_score', 'communication_score', 'leadership_score', 'problem_solving_score', 'attendance_rate', 'conflict_rate']
        for col in score_columns:
            df[col] = df[col].clip(lower=0, upper=100)

        # 3. FEATURE ENGINEERING (Bắt buộc phải có để AI phân tích)
        df['skill_gap'] = df['skill_level'] - df['required_skill_level']
        df['hours_per_day'] = df['workload_hours'] / (df['deadline_days'] + 1e-5)
        df['avg_soft_skill'] = df[['communication_score', 'leadership_score', 'problem_solving_score']].mean(axis=1)

        # 4. SẮP XẾP LẠI CỘT CHO ĐÚNG THỨ TỰ LÚC TRAIN (Rất quan trọng!)
        feature_columns = [
            'experience_years', 'education_level', 'skill_level', 
            'technical_skill_score', 'communication_score', 'leadership_score', 'problem_solving_score', 
            'task_complexity', 'required_skill_level', 'deadline_days', 'workload_hours', 'task_priority', 
            'team_size', 'attendance_rate', 'performance_rating', 'conflict_rate', 
            'skill_gap', 'hours_per_day', 'avg_soft_skill'
        ]
        df = df[feature_columns]

        # 5. SCALE DỮ LIỆU (Đưa về Min-Max 0-1)
        X_scaled = scaler.transform(df)

        # 6. DỰ ĐOÁN KẾT QUẢ
        predicted_class_idx = model.predict(X_scaled)[0]
        
        # 7. MAP TỪ SỐ (0,1,2) VỀ LẠI CHỮ BẰNG LABEL ENCODER
        if label_encoder is not None:
            predicted_status = label_encoder.inverse_transform([predicted_class_idx])[0]
        else:
            predicted_status = str(predicted_class_idx)

        # 8. TRẢ VỀ CHO C#
        return {
            "is_success": True,
            "prediction": predicted_status,
            "prediction_code": int(predicted_class_idx),
            "analyzed_metrics": {
                "skill_gap": float(df['skill_gap'].iloc[0]),
                "hours_per_day": float(df['hours_per_day'].iloc[0])
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))