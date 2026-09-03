"""FastAPI service for the three persisted assignment pipelines."""

from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

PIPELINES: dict[str, object] = {}
METADATA: dict[str, dict] = {}
DATASETS: dict[str, pd.DataFrame] = {}


def _load_json(filename: str) -> dict:
    return json.loads((MODEL_DIR / filename).read_text(encoding="utf-8"))


def load_assets() -> None:
    """Load artifacts from paths independent of the process working directory."""
    PIPELINES.update(
        diabetes=joblib.load(MODEL_DIR / "diabetes_pipeline.joblib"),
        housing=joblib.load(MODEL_DIR / "housing_pipeline.joblib"),
        ecommerce=joblib.load(MODEL_DIR / "ecommerce_pipeline.joblib"),
    )
    METADATA.update(
        diabetes=_load_json("diabetes_metadata.json"),
        housing=_load_json("housing_metadata.json"),
        ecommerce=_load_json("ecommerce_metadata.json"),
    )
    DATASETS.update(
        diabetes=pd.read_csv(BASE_DIR / "diabetes_dataset.csv"),
        housing=pd.read_csv(BASE_DIR / "VN_housing_dataset.csv"),
        ecommerce=pd.read_csv(BASE_DIR / "Womens Clothing E-Commerce Reviews.csv"),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_assets()
    except Exception as exc:  # pragma: no cover - deployment diagnostic path
        PIPELINES.clear()
        METADATA["startup_error"] = {"detail": str(exc)}
    yield


app = FastAPI(
    title="ASG 02 — Intelligent Systems API",
    version="3.0.0",
    description="Deployment-consistent sklearn pipelines for diabetes, housing, and text-enhanced e-commerce interest discovery.",
    lifespan=lifespan,
)

default_origins = "http://localhost:8000,http://127.0.0.1:8000"
allowed_origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", default_origins).split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DiabetesInput(StrictInput):
    age: float = Field(ge=0, le=120)
    bmi: float = Field(ge=10, le=80)
    hbA1c_level: float = Field(ge=2, le=20)
    blood_glucose_level: float = Field(ge=40, le=500)
    gender: Literal["Female", "Male", "Other"]
    smoking_history: Literal["No Info", "never", "former", "current", "not current", "ever"]


class HousingInput(StrictInput):
    dien_tich: float = Field(ge=10, le=1000)
    so_phong_ngu: float = Field(ge=0, le=30)
    so_tang: float = Field(ge=0, le=30)
    quan: str = Field(min_length=2, max_length=100)
    loai_hinh: str = Field(min_length=2, max_length=100)
    phap_ly: str = Field(min_length=2, max_length=100)


class EcommerceInput(StrictInput):
    title: str = Field(default="", max_length=200)
    review_text: str = Field(min_length=10, max_length=3000)
    age: int = Field(ge=18, le=99)
    rating: int = Field(ge=1, le=5)
    recommended_ind: bool
    positive_feedback_count: int = Field(ge=0, le=10_000)


def require_model(name: str):
    if name not in PIPELINES:
        detail = METADATA.get("startup_error", {}).get("detail", "model not loaded")
        raise HTTPException(status_code=503, detail=f"{name} pipeline unavailable: {detail}")
    return PIPELINES[name]


def _first_number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+(?:[\.,]\d+)?", str(value))
    return float(match.group(0).replace(",", ".")) if match else np.nan


@app.get("/", include_in_schema=False)
def web_client():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok" if len(PIPELINES) == 3 else "degraded",
        "loaded_pipelines": sorted(PIPELINES),
        "version": app.version,
    }


@app.get("/model-info")
def model_info():
    return {name: METADATA.get(name, {}) for name in ("diabetes", "housing", "ecommerce")}


@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    pipeline = require_model("diabetes")
    frame = pd.DataFrame([data.model_dump()])
    prediction = int(pipeline.predict(frame)[0])
    probabilities = pipeline.predict_proba(frame)[0]
    classes = list(pipeline.classes_)
    risk_probability = float(probabilities[classes.index(1)])
    predicted_confidence = float(probabilities[classes.index(prediction)])

    similar_cases = []
    source = DATASETS.get("diabetes", pd.DataFrame())
    if not source.empty:
        matches = source[(source["gender"] == data.gender) & ((source["age"] - data.age).abs() <= 5)].copy()
        matches["distance"] = (matches["age"] - data.age).abs() + (matches["bmi"] - data.bmi).abs() / 5
        for _, row in matches.nsmallest(3, "distance").iterrows():
            similar_cases.append({
                "Tuổi": float(row["age"]),
                "BMI": float(row["bmi"]),
                "Glucose": float(row["blood_glucose_level"]),
                "Nhãn thực tế": "Tiểu đường" if int(row["diabetes"]) == 1 else "Không tiểu đường",
            })

    return {
        "prediction": "Cảnh báo nguy cơ tiểu đường" if prediction == 1 else "Không phát hiện nguy cơ",
        "predicted_class": prediction,
        "confidence": round(predicted_confidence * 100, 2),
        "risk_probability": round(risk_probability * 100, 2),
        "disclaimer": "Kết quả hỗ trợ sàng lọc, không thay thế chẩn đoán y khoa.",
        "similar_cases": similar_cases,
    }


@app.post("/predict/housing")
def predict_housing(data: HousingInput):
    pipeline = require_model("housing")
    frame = pd.DataFrame([{
        "Diện tích": data.dien_tich,
        "Số phòng ngủ": data.so_phong_ngu,
        "Số tầng": data.so_tang,
        "Quận": data.quan,
        "Loại hình nhà ở": data.loai_hinh,
        "Giấy tờ pháp lý": data.phap_ly,
    }])
    prediction = max(0.0, float(pipeline.predict(frame)[0]))
    q90 = float(METADATA["housing"].get("validation_absolute_error_q90", 0.0))

    similar_cases = []
    source = DATASETS.get("housing", pd.DataFrame())
    if not source.empty:
        matches = source[(source["Quận"] == data.quan) & (source["Loại hình nhà ở"] == data.loai_hinh)].copy()
        matches["area_numeric"] = matches["Diện tích"].map(_first_number)
        matches["distance"] = (matches["area_numeric"] - data.dien_tich).abs()
        for _, row in matches.nsmallest(3, "distance").iterrows():
            similar_cases.append({
                "Khu vực": str(row["Quận"]),
                "Diện tích": str(row.get("Diện tích", "N/A")),
                "Loại": str(row.get("Loại hình nhà ở", "N/A")),
            })

    return {
        "predicted_price": round(prediction, 2),
        "lower_bound": round(max(0.0, prediction - q90), 2),
        "upper_bound": round(prediction + q90, 2),
        "unit": "Tỷ VNĐ",
        "range_method": "Khoảng heuristic ± bách phân vị 90% sai số tuyệt đối trên validation; không phải confidence interval.",
        "similar_cases": similar_cases,
    }


@app.post("/predict/ecommerce")
def predict_ecommerce(data: EcommerceInput):
    pipeline = require_model("ecommerce")
    raw = data.model_dump()
    frame = pd.DataFrame([{
        "Review Combined": f"{raw['title']} {raw['review_text']}".strip(),
        "Age": raw["age"],
        "Rating": raw["rating"],
        "Recommended IND": int(raw["recommended_ind"]),
        "Positive Feedback Count": raw["positive_feedback_count"],
    }])
    prediction = str(pipeline.predict(frame)[0])
    probabilities = pipeline.predict_proba(frame)[0]
    classes = [str(item) for item in pipeline.classes_]
    confidence = float(probabilities[classes.index(prediction)])
    class_probabilities = {
        label: round(float(probability) * 100, 2)
        for label, probability in zip(classes, probabilities)
    }

    similar_cases = []
    source = DATASETS.get("ecommerce", pd.DataFrame())
    if not source.empty:
        filtered = source[source["Department Name"] == prediction].dropna(subset=["Review Text"]).copy()
        filtered["distance"] = (
            (filtered["Rating"] - data.rating).abs()
            + (filtered["Age"] - data.age).abs() / 20
        )
        for _, row in filtered.nsmallest(3, "distance").iterrows():
            title = "" if pd.isna(row.get("Title")) else str(row.get("Title", ""))
            review = str(row["Review Text"]).strip()
            similar_cases.append({
                "Tuổi": int(row["Age"]),
                "Rating": int(row["Rating"]),
                "Tiêu đề": title or "(không có)",
                "Trích review": review[:120] + ("..." if len(review) > 120 else ""),
            })

    return {
        "interest": prediction,
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "class_probabilities": class_probabilities,
        "explanation": "Nhãn là nhóm sản phẩm được suy ra từ nội dung review kết hợp tuổi, rating, khuyến nghị và phản hồi tích cực.",
        "confidence_note": "Confidence là xác suất do mô hình ước lượng, không phải độ chắc chắn thống kê.",
        "similar_cases": similar_cases,
    }
