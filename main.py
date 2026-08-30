from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Intelligent Systems API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LOAD MODELS ---
try:
    rf_diabetes = joblib.load("models/rf_diabetes_model.joblib")
    cols_diabetes = joblib.load("models/diabetes_columns.joblib")
    rf_housing = joblib.load("models/rf_housing_model.joblib")
    cols_housing = joblib.load("models/housing_feature_columns.joblib")
    rf_ecommerce = joblib.load("models/rf_customer_behavior_model.joblib")
    cols_ecommerce = joblib.load("models/customer_behavior_columns.joblib")
except Exception as e:
    print(f"Lỗi tải mô hình: {e}")

# --- 2. LOAD DATASETS CHO TÍNH NĂNG TÌM KIẾM TƯƠNG ĐỒNG ---
try:
    df_diabetes = pd.read_csv("diabetes_dataset.csv")
except:
    df_diabetes = pd.DataFrame()

try:
    df_housing = pd.read_csv("VN_housing_dataset.csv")
except:
    df_housing = pd.DataFrame()

try:
    df_ecommerce = pd.read_csv("E-commerce Customer Behavior.csv")
except:
    df_ecommerce = pd.DataFrame()


# --- 3. SCHEMAS & UTILS ---
class DiabetesInput(BaseModel):
    age: float;
    bmi: float;
    hbA1c_level: float;
    blood_glucose_level: float;
    gender: str;
    smoking_history: str


class HousingInput(BaseModel):
    dien_tich: float;
    so_phong_ngu: float;
    so_tang: float;
    quan: str;
    loai_hinh: str;
    phap_ly: str


class EcommerceInput(BaseModel):
    age: int;
    total_spend: float;
    items_purchased: int;
    average_rating: float;
    days_since_last_purchase: int
    gender: str;
    city: str;
    membership_type: str;
    discount_applied: bool


def preprocess_input(data_dict: dict, saved_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([data_dict])
    df_encoded = pd.get_dummies(df)
    return df_encoded.reindex(columns=saved_columns, fill_value=0)


# --- 4. API ENDPOINTS ---
@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    df_input = preprocess_input(data.dict(), cols_diabetes)
    prediction = rf_diabetes.predict(df_input)[0]
    confidence = max(rf_diabetes.predict_proba(df_input)[0])

    # Tìm kiếm ca bệnh tương đồng (Cùng giới tính, tuổi +- 5)
    similar_cases = []
    if not df_diabetes.empty:
        matches = df_diabetes[
            (df_diabetes['gender'] == data.gender) &
            (abs(df_diabetes['age'] - data.age) <= 5)
            ].head(3)
        for _, row in matches.iterrows():
            stt = "Tiểu đường" if row.get('diabetes', 0) == 1 else "Khỏe mạnh"
            similar_cases.append(
                {"Tuổi": row['age'], "BMI": row['bmi'], "Glucose": row['blood_glucose_level'], "Thực tế": stt})

    result = "Cảnh báo Tiểu đường" if prediction == 1 else "Khỏe mạnh"
    return {"prediction": result, "confidence": round(confidence * 100, 2), "similar_cases": similar_cases}


@app.post("/predict/housing")
def predict_housing(data: HousingInput):
    raw_dict = {"Diện tích": data.dien_tich, "Số phòng ngủ": data.so_phong_ngu, "Số tầng": data.so_tang,
                "Quận": data.quan, "Loại hình nhà ở": data.loai_hinh, "Giấy tờ pháp lý": data.phap_ly}
    df_input = preprocess_input(raw_dict, cols_housing)

    prediction = rf_housing.predict(df_input)[0]
    all_preds = [tree.predict(df_input.values)[0] for tree in rf_housing.estimators_]
    std_dev = np.std(all_preds)

    # Tìm kiếm nhà tương đồng (Cùng quận, cùng loại hình)
    similar_cases = []
    if not df_housing.empty:
        matches = df_housing[
            (df_housing['Quận'] == data.quan) & (df_housing['Loại hình nhà ở'] == data.loai_hinh)
            ].head(3)
        for _, row in matches.iterrows():
            similar_cases.append({"Khu vực": row['Quận'], "Diện tích": row.get('Diện tích', 'N/A'),
                                  "Loại": row.get('Loại hình nhà ở', 'N/A')})

    return {"predicted_price": round(prediction, 2), "lower_bound": max(0, round(prediction - std_dev, 2)),
            "upper_bound": round(prediction + std_dev, 2), "unit": "Tỷ VNĐ", "similar_cases": similar_cases}


@app.post("/predict/ecommerce")
def predict_ecommerce(data: EcommerceInput):
    raw_dict = {"Age": data.age, "Total Spend": data.total_spend, "Items Purchased": data.items_purchased,
                "Average Rating": data.average_rating,
                "Days Since Last Purchase": data.days_since_last_purchase, "Gender": data.gender, "City": data.city,
                "Membership Type": data.membership_type, "Discount Applied": data.discount_applied}
    df_input = preprocess_input(raw_dict, cols_ecommerce)

    prediction = rf_ecommerce.predict(df_input)[0]
    confidence = max(rf_ecommerce.predict_proba(df_input)[0])

    # Tìm khách hàng tương đồng (Cùng hạng thẻ, tìm người có mức chi tiêu gần nhất)
    similar_cases = []
    if not df_ecommerce.empty:
        filtered = df_ecommerce[df_ecommerce['Membership Type'] == data.membership_type].copy()
        filtered['diff'] = abs(filtered['Total Spend'] - data.total_spend)
        matches = filtered.sort_values('diff').head(3)
        for _, row in matches.iterrows():
            similar_cases.append(
                {"Tuổi": row['Age'], "Chi tiêu": f"${row['Total Spend']}", "Hài lòng": row['Satisfaction Level']})

    return {"prediction": prediction, "confidence": round(confidence * 100, 2), "similar_cases": similar_cases}
