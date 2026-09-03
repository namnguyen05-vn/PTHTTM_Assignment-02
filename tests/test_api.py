from fastapi.testclient import TestClient

from main import app


def test_health_and_frontend():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["loaded_pipelines"] == ["diabetes", "ecommerce", "housing"]
        assert client.get("/").status_code == 200


def test_all_prediction_endpoints():
    with TestClient(app) as client:
        diabetes = client.post("/predict/diabetes", json={
            "age": 45, "bmi": 28.5, "hbA1c_level": 6.5,
            "blood_glucose_level": 150, "gender": "Male", "smoking_history": "never",
        })
        assert diabetes.status_code == 200
        assert 0 <= diabetes.json()["risk_probability"] <= 100

        housing = client.post("/predict/housing", json={
            "dien_tich": 50, "so_phong_ngu": 2, "so_tang": 3,
            "quan": "Quận Cầu Giấy", "loai_hinh": "Nhà ngõ, hẻm", "phap_ly": "Đã có sổ",
        })
        assert housing.status_code == 200
        assert housing.json()["predicted_price"] >= 0
        assert "không phải confidence interval" in housing.json()["range_method"]

        ecommerce = client.post("/predict/ecommerce", json={
            "title": "Perfect summer dress",
            "review_text": "This dress is light, comfortable and fits beautifully for summer.",
            "age": 30, "rating": 5, "recommended_ind": True,
            "positive_feedback_count": 3,
        })
        assert ecommerce.status_code == 200
        assert ecommerce.json()["interest"] in {"Bottoms", "Dresses", "Intimate", "Jackets", "Tops", "Trend"}
        assert 0 <= ecommerce.json()["confidence"] <= 100
        assert set(ecommerce.json()["class_probabilities"]) == {
            "Bottoms", "Dresses", "Intimate", "Jackets", "Tops", "Trend"
        }


def test_input_validation_rejects_bad_values_and_extra_fields():
    with TestClient(app) as client:
        response = client.post("/predict/diabetes", json={
            "age": -1, "bmi": 28.5, "hbA1c_level": 6.5,
            "blood_glucose_level": 150, "gender": "Male", "smoking_history": "never",
            "unexpected": "field",
        })
        assert response.status_code == 422

        bad_review = client.post("/predict/ecommerce", json={
            "review_text": "short",
            "age": 30, "rating": 6, "recommended_ind": True,
            "positive_feedback_count": 0,
        })
        assert bad_review.status_code == 422
