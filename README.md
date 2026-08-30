# HỆ THỐNG TRÍ TUỆ NHÂN TẠO ĐA ỨNG DỤNG (Multi-Domain Intelligent System)

**Môn học:** Phát triển Hệ thống Thông minh  
**Sinh viên thực hiện:** Nguyễn Ngọc Hoàng Nam (B23DCCN585)  

Dự án này là một hệ thống AI đa nhiệm được thiết kế theo kiến trúc Client-Server, cung cấp 3 dịch vụ dự đoán tích hợp Suy luận dựa trên ca mẫu (Case-based Reasoning - XAI):
1. **Y tế:** Chẩn đoán rủi ro mắc bệnh Tiểu đường (Phân loại nhị phân).
2. **Bất động sản:** Định giá nhà và tính toán khoảng giá dao động (Hồi quy).
3. **Thương mại:** Dự đoán hành vi và mức độ hài lòng của khách hàng E-commerce (Phân loại đa lớp).

## 🛠 Công nghệ sử dụng
*   **Machine Learning:** Scikit-Learn (Random Forest), Pandas, Numpy, SMOTE.
*   **Backend API:** Python FastAPI, Uvicorn, Pydantic (Đảm bảo Data Alignment, chống rò rỉ dữ liệu).
*   **Frontend:** HTML5, JavaScript (Fetch API), Bootstrap 5 (Giao diện Glassmorphism Responsive).

## 📂 Cấu trúc thư mục
```text
PTHTTM_Assignment-02/
│
├── models/                              # Chứa các file mô hình Random Forest (.joblib) đã được huấn luyện
├── ASG_02_1.ipynb                       # Notebook xử lý & Huấn luyện mô hình Y tế
├── ASG_02_2.ipynb                       # Notebook xử lý & Huấn luyện mô hình Giá nhà
├── ASG_02_3.ipynb                       # Notebook xử lý & Huấn luyện mô hình E-commerce
├── diabetes_dataset.csv                 # Dữ liệu gốc để truy xuất ca mẫu y tế
├── VN_housing_dataset.csv               # Dữ liệu gốc để truy xuất ca mẫu giá nhà
├── E-commerce Customer Behavior.csv     # Dữ liệu gốc để truy xuất ca mẫu thương mại
├── main.py                              # Mã nguồn Backend API (Xử lý suy luận & XAI)
├── index.html                           # Mã nguồn Giao diện người dùng
├── requirements.txt                     # Danh sách các thư viện Python cần thiết
└── README.md                            # Tài liệu hướng dẫn cài đặt
```

## 🚀 Hướng dẫn Cài đặt và Vận hành

Hệ thống yêu cầu chạy song song 2 máy chủ: Một máy chủ AI Backend (cổng 8000) và một máy chủ Web Frontend (cổng 9000).

### Bước 1: Cài đặt Môi trường (Environment Setup)
Yêu cầu hệ thống đã cài đặt **Python 3.9+**. Mở Terminal hoặc Command Prompt và chạy lệnh sau để cài đặt toàn bộ các thư viện lõi:
```bash
pip install -r requirements.txt
```

### Bước 2: Khởi động Máy chủ AI (Backend API)
Mở một cửa sổ Terminal mới, di chuyển đến thư mục chứa file `main.py`. Khởi động FastAPI Backend với cờ `--host 0.0.0.0` để cho phép các thiết bị trong mạng LAN kết nối:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Bước 3: Khởi động Giao diện Người dùng (Web Server)
Giữ nguyên cửa sổ Terminal của Backend. Mở một **cửa sổ Terminal thứ hai** tại cùng thư mục dự án và khởi động máy chủ Web tĩnh:
```bash
python -m http.server 9000
```

### Bước 4: Trải nghiệm Hệ thống (Demo)
*   **Chạy trên Laptop (Local):** Mở trình duyệt Web (Chrome, Edge, Safari) và truy cập vào đường dẫn:  
    👉 `http://localhost:9000`
*   **Chạy trên Điện thoại (Mobile):** 
    1. Lấy địa chỉ IPv4 của Laptop (vd: `192.168.1.15`).
    2. Mở file `index.html`, tìm biến `const API_URL` và đổi thành: `const API_URL = "http://192.168.1.15:8000";`
    3. Đảm bảo điện thoại và Laptop kết nối chung mạng Wi-Fi (hoặc tắt Tường lửa Windows nếu bị chặn).
    4. Mở trình duyệt trên điện thoại và truy cập: `http://192.168.1.15:9000`

---
*Dự án được xây dựng tuân thủ quy trình chuẩn CRISP-DM và áp dụng các kỹ thuật ngăn chặn rò rỉ dữ liệu (Data Leakage) nghiêm ngặt để đảm bảo độ tin cậy khi triển khai.*
