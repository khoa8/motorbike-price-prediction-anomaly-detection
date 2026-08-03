# Motorbike Price Prediction & Price Anomaly Detection

Hệ thống dự đoán giá xe máy cũ và phát hiện các tin có giá bất thường trên dữ liệu tin đăng tại TP.HCM. Project kết hợp:

- **scikit-learn** cho mô hình Machine Learning truyền thống;
- **Apache Spark ML** cho phần benchmark Big Data / Distributed Machine Learning;
- **Streamlit** để triển khai ứng dụng tương tác.

> **Trạng thái project:** phiên bản hiện tại được hoàn thiện cho buổi thuyết trình và demo. Sau buổi demo, repository sẽ tiếp tục được chuẩn hóa sang tiếng Anh, tinh gọn notebook, bổ sung test và hoàn thiện tài liệu theo chuẩn portfolio.

## Live Demo

**Streamlit application:**  
https://motorbike-price-analytics.streamlit.app/

Ứng dụng gồm ba chức năng chính:

1. **Price Prediction** — dự đoán mức giá tham khảo cho một xe.
2. **Anomaly Check** — đánh giá một giá đăng cụ thể và giải thích các tín hiệu bất thường.
3. **Batch Check** — xử lý nhiều tin cùng lúc bằng file CSV.

## Project Overview

Project giải quyết hai bài toán liên quan nhưng khác nhau:

### 1. Price Prediction

Dự đoán giá xe theo các đặc điểm như:

- thương hiệu và dòng xe;
- năm đăng ký, tuổi xe và số kilomet;
- loại xe, dung tích và xuất xứ;
- quận/huyện;
- độ dài tiêu đề và mô tả;
- phân khúc xe;
- một số engineered features phục vụ deployment.

### 2. Price Anomaly Detection

Phát hiện các tin có giá quá thấp hoặc quá cao bằng bốn tín hiệu thành phần:

1. **Residual-Z** theo phân khúc;
2. giá nằm ngoài **P1–P99**;
3. giá nằm ngoài **P10–P90**;
4. **Isolation Forest**.

Bốn tín hiệu được kết hợp thành `anomaly_score`. Ngưỡng cảnh báo được hiệu chỉnh theo top khoảng 5% điểm anomaly trong dữ liệu lịch sử.

Kết quả chỉ dùng để **hỗ trợ định giá và ưu tiên kiểm duyệt**. Một tin bị gắn cờ không đồng nghĩa chắc chắn nhập sai hoặc gian lận.

## Dataset

Dữ liệu gồm các tin đăng xe máy cũ tại TP.HCM trước ngày **01/07/2025**.

| Nội dung | Số lượng |
|---|---:|
| Dòng dữ liệu gốc | 7,208 |
| Dòng dùng huấn luyện regression | 7,141 |
| Dòng có giá dương dùng cho anomaly detection | 7,193 |
| Số phân khúc | 93 |

Dữ liệu regression loại các mức giá nhỏ hơn 1 triệu hoặc lớn hơn 1 tỷ đồng để tránh huấn luyện trên các trường hợp có khả năng là lỗi nhập hoặc cực trị. Các dòng có giá dương này vẫn được giữ trong tập anomaly detection.

> **Data usage notice:** dữ liệu gốc không được đưa vào repository. Dataset chỉ được sử dụng cho mục đích học tập và nghiên cứu; cần kiểm tra quyền sở hữu và điều khoản sử dụng trước khi tái phân phối hoặc sử dụng thương mại.

## Modeling Design

### Data preparation

- Chuyển giá về đơn vị triệu đồng.
- Tạo `age = 2025 - year`.
- Xử lý missing values và duplicate.
- Trích xuất quận/huyện.
- Tạo độ dài tiêu đề và mô tả.
- Tạo 93 phân khúc theo thương hiệu, dòng xe, năm, loại xe và dung tích.
- Dùng `log1p(price)` khi huấn luyện và `expm1` khi đưa prediction về đơn vị triệu đồng.

### Train/test

Regression sử dụng 7,141 dòng:

| Tập | Số dòng |
|---|---:|
| Train | 5,712 |
| Test | 1,429 |

Train/test được dùng chung cho benchmark scikit-learn và Spark. Metric chính để lựa chọn model là **MAE**.

### Out-of-fold prediction

Anomaly detection sử dụng **5-fold out-of-fold prediction** để mỗi dòng được dự đoán bởi một model không học trực tiếp từ chính dòng đó. Cách này giúp residual phản ánh sai số thực tế hơn so với prediction in-sample.

## Key Results

### Benchmark notebook

| Environment | Selected model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| scikit-learn | Random Forest | **10.129** | 33.262 | 0.465 |
| Spark ML | Random Forest | **10.543** | 33.431 | 0.460 |

Random Forest được chọn trong từng môi trường vì có MAE thấp nhất. Linear Regression có RMSE và R² tốt hơn trong một số thử nghiệm, cho thấy không có một model tốt nhất trên mọi metric.

### Deployment model

Ứng dụng Streamlit không dùng nguyên trạng benchmark Random Forest. Deployment sử dụng **Premium-aware Random Forest**, bổ sung engineered features và sample weights để cải thiện nhóm xe giá cao.

#### Test-set evaluation

| Model | MAE | RMSE | R² | Premium MAE |
|---|---:|---:|---:|---:|
| Benchmark Random Forest | 10.129 | 33.262 | 0.465 | 89.764 |
| Premium-aware Random Forest | **9.773** | **31.649** | **0.516** | **80.602** |

#### 5-fold OOF evaluation

| Model | MAE | RMSE | R² | Premium MAE |
|---|---:|---:|---:|---:|
| Benchmark RF | 10.110 | 32.089 | 0.491 | 89.341 |
| Premium-aware RF | **9.741** | **30.589** | **0.538** | **81.032** |

So với benchmark OOF:

- overall MAE giảm khoảng **3.66%**;
- premium MAE giảm khoảng **9.30%**;
- premium RMSE giảm khoảng **5.52%**.

### Deployment anomaly calibration

| Nội dung | Giá trị |
|---|---:|
| Anomaly threshold | 48.138 |
| Tổng tin trong top anomaly | 360 |
| Tin quá rẻ | 114 |
| Tin quá đắt | 246 |

### Original scikit-learn and Spark anomaly benchmark

| Nội dung | scikit-learn | Spark |
|---|---:|---:|
| Tin bị gắn cờ | 360 | 367 |
| Quá rẻ | 108 | 108 |
| Quá đắt | 252 | 259 |

Hai môi trường cùng cảnh báo 307 tin. Agreement rate là 98.43%, trong khi Jaccard similarity trên riêng tập anomaly là 0.731.

> **Lưu ý:** các số liệu benchmark và deployment thuộc hai giai đoạn khác nhau. Streamlit sử dụng deployment artifacts và ngưỡng 48.138, không sử dụng trực tiếp toàn bộ kết quả anomaly benchmark ban đầu.

## Streamlit Application

### Price Prediction

Người dùng nhập thông tin xe và nhận:

- giá dự đoán;
- phân khúc;
- P10 và P90 của phân khúc;
- cảnh báo khi xe thuộc nhóm hiếm hoặc nhóm giá cao.

### Anomaly Check

Ngoài giá dự đoán, ứng dụng hiển thị:

- giá đăng và mức chênh lệch;
- anomaly score và threshold;
- trạng thái **Bình thường**, **Cần kiểm tra thủ công** hoặc **Bất thường**;
- từng cờ Residual-Z, P1–P99, P10–P90 và Isolation Forest;
- lý do và điểm đóng góp của từng tín hiệu.

### Batch Check

Người dùng có thể tải CSV để:

- dự đoán giá cho nhiều dòng;
- chấm anomaly cho các dòng có `listed_price_million`;
- lọc theo trạng thái;
- tải kết quả và danh sách dòng lỗi;
- xem top anomaly score và phân bố trạng thái.

File mẫu có sẵn trong `examples/motorbike_batch_template.csv`.

## Repository Structure

```text
motorbike-price-prediction-anomaly-detection/
├── .streamlit/
│   └── config.toml
├── app.py
├── assets/
│   └── motorbike_banner.png
├── artifacts/
│   ├── price_model.joblib
│   ├── isolation_preprocessor.joblib
│   ├── isolation_forest.joblib
│   ├── deployment_config.json
│   ├── deployment_results_summary.json
│   ├── segment_rules.json
│   └── segment_statistics.csv
├── docs/
│   └── README_project2.md
├── examples/
│   └── motorbike_batch_template.csv
├── notebooks/
│   ├── project2_motorbike_price_anomaly.ipynb
│   └── project2_motorbike_price_anomaly_streamlit.ipynb
├── reports/
│   ├── project2_results_summary.json
│   ├── sklearn_model_comparison.csv
│   ├── spark_model_comparison.csv
│   ├── sklearn_feature_importance.csv
│   └── spark_rf_feature_importance.csv
├── src/
│   ├── batch.py
│   ├── features.py
│   └── inference.py
├── check_project.py
├── generate_requirements.py
├── requirements.txt
├── README_SETUP.md
└── README.md
```

### Notebook roles

- `project2_motorbike_price_anomaly.ipynb`: notebook gốc cho benchmark scikit-learn, Spark và anomaly detection.
- `project2_motorbike_price_anomaly_streamlit.ipynb`: mở rộng notebook gốc bằng phần feature engineering, lựa chọn deployment model, hiệu chỉnh anomaly và xuất Streamlit artifacts.

## Run Locally

### 1. Clone repository

```bash
git clone https://github.com/khoa8/motorbike-price-prediction-anomaly-detection.git
cd motorbike-price-prediction-anomaly-detection
```

### 2. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Trên Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Validate project

```bash
python -m compileall app.py src
python check_project.py
```

### 4. Start Streamlit

```bash
python -m streamlit run app.py
```

Mở `http://localhost:8501`.

## Reproduce the Notebooks

Notebook được thiết kế để chạy trên Google Colab.

1. Mở notebook trong thư mục `notebooks/`.
2. Đặt `data_motobikes.xlsx` trong `/content`, hoặc upload khi notebook yêu cầu.
3. Chọn `Runtime -> Run all`.
4. Tải các output từ `/content/project2_outputs/`.

Runtime được ghi nhận trong lần chạy hiện tại:

```text
Python 3.12.13
Java 17
PySpark 4.0.3
```

Phiên bản runtime thực tế có thể thay đổi theo môi trường Colab.

## Limitations

- Dataset không có ground-truth anomaly, nên chưa thể báo cáo Precision, Recall hoặc F1 thực sự.
- Anomaly weights và top-5% threshold là heuristic.
- Các phân khúc hiếm và xe trên 100 triệu có ít mẫu và sai số cao hơn.
- P1/P99 có thể chưa ổn định trong các phân khúc nhỏ.
- Dữ liệu chỉ là snapshot tại TP.HCM trước ngày 01/07/2025.
- Mô hình cần được theo dõi và huấn luyện lại khi thị trường thay đổi.
- Ứng dụng hỗ trợ quyết định, không thay thế thẩm định của con người.

## Planned Improvements

- Chuẩn hóa toàn bộ tài liệu và giao diện sang tiếng Anh.
- Tinh gọn notebook và tách code tái sử dụng thành module.
- Bổ sung unit tests, smoke tests và CI.
- Theo dõi data drift và model drift.
- Thu thập thêm dữ liệu xe cao cấp, xe hiếm và xe phân khối lớn.
- Tối ưu hyperparameters và anomaly threshold bằng dữ liệu có nhãn.
- Bổ sung model card, data statement và tài liệu kiến trúc.
- Hoàn thiện licensing cho code, dữ liệu và model artifacts trước khi public rộng rãi.

## Team

**Nhóm 2**

- Nguyễn Minh Khoa
- Nguyễn Hoàng Quỳnh Anh
