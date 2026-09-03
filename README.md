# Assignment 02 — Intelligent Systems Development

Repository chính thức: [namnguyen05-vn/PTHTTM_Assignment-02](https://github.com/namnguyen05-vn/PTHTTM_Assignment-02)

Hệ thống gồm ba pipeline học máy có thể tái lập và triển khai qua một FastAPI cùng giao diện web responsive:

1. dự đoán nguy cơ tiểu đường (binary classification);
2. định giá nhà Hà Nội (regression);
3. khám phá nhóm sản phẩm khách hàng quan tâm từ review e-commerce (multiclass text classification).

## Điểm kỹ thuật chính

- Bản ghi trùng được xử lý trước khi chia dữ liệu.
- Chia train/validation/test xấp xỉ 70/15/15; e-commerce dùng stratified group split theo `Clothing ID`, housing dùng group-aware split để các profile/sản phẩm không nằm ở cả train và test.
- Imputation, scaling, one-hot encoding và TF-IDF đều nằm trong `sklearn.pipeline.Pipeline` và chỉ được fit trên train khi so sánh mô hình.
- Lựa chọn mô hình bằng validation; test chỉ dùng một lần sau khi chốt mô hình.
- API nhận đúng các raw features mà pipeline đã huấn luyện; không fit encoder/scaler lại khi inference.

## Dữ liệu

| Ứng dụng | Dataset | Dữ liệu gốc | Target |
|---|---|---:|---|
| Diabetes | [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset/data) | 100,000 × 16 | `diabetes` |
| Housing | [Vietnam Housing Dataset](https://www.kaggle.com/code/kerneler/starter-vietnam-housing-dataset-5742ed64-7/input) | 82,497 × 13 | `Price` được tạo từ diện tích × giá/m² |
| E-commerce | [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) | 23,486 × 11 | `Department Name` |

Ứng dụng e-commerce kết hợp `Title + Review Text` với `Age`, `Rating`, `Recommended IND` và `Positive Feedback Count`. `Division Name` và `Class Name` bị loại vì có thể làm lộ target; `Clothing ID` chỉ dùng để chia group và không đi vào mô hình.

Notebook minh họa đầy đủ chuỗi `Comment → Tokens → Token IDs [B,T] → Embeddings [B,T,d]` với `B=4`, `T=32`, `d=32`. Pipeline triển khai sử dụng TF-IDF unigram/bigram kết hợp bốn biến tabular, tạo đầu vào sparse `B × 8,004`.

## Cấu trúc quan trọng

```text
ASG_02_1.ipynb               # diabetes: 5 mô hình
ASG_02_2.ipynb               # housing: 5 mô hình
ASG_02_3.ipynb               # e-commerce: 6 mô hình
Womens Clothing E-Commerce Reviews.csv  # tải theo hướng dẫn bên dưới
models/*_pipeline.joblib     # ba pipeline hoàn chỉnh, được tạo sau khi huấn luyện
models/*_metadata.json       # đặc trưng, split, model, test metrics
artifacts/*_metrics.csv      # bảng so sánh validation
artifacts/ecommerce_representation_ablation.csv
figures/*.png                # EDA và đánh giá test
main.py                      # FastAPI + validation + static client
index.html                   # responsive web/mobile interface
demo/                        # ảnh minh họa giao diện desktop và mobile-responsive
tests/test_api.py            # smoke tests cho API và validation
```

## Cài đặt và chạy

Yêu cầu Python 3.11 trở lên.

Với bản clone từ GitHub, tải dataset [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews), giải nén và đặt file `Womens Clothing E-Commerce Reviews.csv` tại thư mục gốc dự án. Bản ZIP bàn giao kèm bài đã chứa sẵn dataset này.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Sau khi chuẩn bị dữ liệu, nếu thư mục `models/` chưa có các file `*_pipeline.joblib`, hãy chạy ba notebook từ đầu đến cuối trước khi khởi động API. Có thể thực hiện bằng lệnh:

```bash
jupyter nbconvert --to notebook --execute --inplace ASG_02_1.ipynb
jupyter nbconvert --to notebook --execute --inplace ASG_02_2.ipynb
jupyter nbconvert --to notebook --execute --inplace ASG_02_3.ipynb
```

Mở:

- giao diện responsive: `http://127.0.0.1:8000/`
- Swagger UI: `http://127.0.0.1:8000/docs`
- health check: `http://127.0.0.1:8000/health`

Không cần sửa IP trong `index.html`: client dùng cùng origin với API. Nếu tách frontend và backend, đặt danh sách origin hợp lệ qua biến môi trường `ALLOWED_ORIGINS` (phân tách bằng dấu phẩy).

## Ví dụ API

```bash
curl -X POST http://127.0.0.1:8000/predict/diabetes \
  -H 'Content-Type: application/json' \
  -d '{"age":45,"bmi":28.5,"hbA1c_level":6.5,"blood_glucose_level":150,"gender":"Male","smoking_history":"never"}'
```

Các endpoint dự đoán:

- `POST /predict/diabetes`
- `POST /predict/housing`
- `POST /predict/ecommerce`

Ví dụ e-commerce:

```bash
curl -X POST http://127.0.0.1:8000/predict/ecommerce \
  -H 'Content-Type: application/json' \
  -d '{"title":"Perfect summer dress","review_text":"This dress is light, comfortable and fits beautifully for summer.","age":30,"rating":5,"recommended_ind":true,"positive_feedback_count":3}'
```

Response gồm `interest`, `confidence`, xác suất của sáu nhóm và các review tham khảo cùng nhóm. `confidence` là xác suất do mô hình ước lượng, không phải độ chắc chắn thống kê.

## Kết quả test cuối

| Ứng dụng | Mô hình chốt | Metric chính | Kết quả |
|---|---|---|---:|
| Diabetes | Random Forest | F1 lớp bệnh | 0.7657 |
| Diabetes | Random Forest | ROC-AUC | 0.9689 |
| Housing | Gradient Boosting | MAE | 1.8184 tỷ VNĐ |
| Housing | Gradient Boosting | R² | 0.5174 |
| E-commerce | Logistic Regression | Accuracy | 0.8042 |
| E-commerce | Logistic Regression | Macro-F1 | 0.5997 |
| E-commerce | Logistic Regression | ROC-AUC OVR | 0.8952 |

Ở validation, Logistic Regression tabular-only chỉ đạt macro-F1 `0.0947`; khi thêm review TF-IDF đạt `0.6007`, tăng `0.5060`. Macro-F1 thấp hơn Accuracy do lớp `Trend` rất hiếm (118/22.623 review sạch), vì vậy không nên chỉ báo cáo Accuracy.

Khoảng giá nhà mà API trả về là khoảng heuristic dựa trên bách phân vị 90% của sai số tuyệt đối trên validation, **không phải confidence interval thống kê**. Kết quả tiểu đường chỉ phục vụ minh họa/sàng lọc, không thay thế chẩn đoán y khoa.

## Chạy kiểm thử

```bash
pytest -q
```

## Tái huấn luyện

Chạy `python tools/build_notebooks.py` để tạo lại ba notebook, sau đó mở và chạy lần lượt từ đầu đến cuối. Mỗi notebook tự tạo lại pipeline, metadata, bảng metric và hình tương ứng. `random_state=42` được cố định để tái lập split và mô hình.

Giao diện responsive là mobile web client: người dùng có thể nhập dữ liệu, gửi REST request và xem prediction/confidence trên điện thoại. Theo phạm vi thực hiện của bài này, responsive web được dùng thay cho native mobile client.

## Ảnh demo

Ảnh minh họa giao diện được lưu trong thư mục [`demo/`](demo/), gồm luồng nhập liệu và kết quả trên desktop/mobile-responsive cho bài toán tiểu đường và định giá bất động sản. Giao diện e-commerce nhận `Title + Review Text` cùng bốn trường tabular và trả về nhóm `Department Name`.
