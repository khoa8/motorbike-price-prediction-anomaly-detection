# Motorbike Price Prediction and Price Anomaly Detection

Dự án xây dựng hệ thống **dự đoán giá đăng bán xe máy cũ** và **phát hiện các tin có giá quá rẻ hoặc quá đắt** trên dữ liệu tin đăng Chợ Tốt tại TP.HCM.

Project được triển khai trên hai môi trường:

- **Machine Learning truyền thống:** scikit-learn.
- **Big Data / Distributed Machine Learning:** Apache Spark ML.

Notebook thực hiện đầy đủ quy trình từ Business Understanding, Data Understanding, Data Preparation, EDA, Modeling, Evaluation đến xuất kết quả.

---

## 1. Business Objectives

Hệ thống hỗ trợ hai nhóm người dùng:

### Người bán

- Nhận mức giá gợi ý dựa trên thông tin xe.
- So sánh giá mong muốn với giá mô hình dự đoán.
- Cân nhắc điều chỉnh giá trước khi đăng tin.

### Nền tảng và người kiểm duyệt

- Phát hiện các tin có giá nghi ngờ quá rẻ hoặc quá đắt.
- Xem từng tín hiệu và lý do khiến tin bị gắn cờ.
- Ưu tiên kiểm tra các tin có anomaly score cao.

Kết quả của mô hình chỉ đóng vai trò **hỗ trợ quyết định**. Hệ thống không tự động kết luận người bán nhập sai và không tự động từ chối tin đăng.

---

## 2. Project Tasks

Project gồm hai bài toán chính.

### Bài toán 1 — Price Prediction

Xây dựng các mô hình regression để dự đoán giá xe trên:

- scikit-learn;
- Spark ML.

Các mô hình được đánh giá bằng:

- MAE;
- RMSE;
- R²;
- Actual vs Predicted;
- Residual Plot;
- Sai số theo nhóm giá và phân khúc;
- Feature Importance.

### Bài toán 2 — Price Anomaly Detection

Phát hiện các xe có giá quá rẻ hoặc quá đắt bằng năm bước/tín hiệu:

1. Residual-Z theo phân khúc.
2. Giá ngoài P1–P99 của phân khúc.
3. Giá ngoài P10–P90 của phân khúc.
4. Unsupervised anomaly detection.
5. Composite anomaly score và ngưỡng top 5%.

---

## 3. Dataset

Dữ liệu gồm các tin đăng xe máy cũ tại TP.HCM trước ngày 01/07/2025.

Một số thuộc tính chính:

- `Giá`;
- `Thương hiệu`;
- `Dòng xe`;
- `Năm đăng ký`;
- `Số Km đã đi`;
- `Loại xe`;
- `Dung tích xe`;
- `Xuất xứ`;
- `Địa chỉ`;
- `Tiêu đề`;
- `Mô tả chi tiết`;
- `Href`.

### Kết quả kiểm tra và làm sạch

| Nội dung | Số lượng |
|---|---:|
| Dòng dữ liệu gốc | 7.208 |
| Số cột | 18 |
| `Href` trùng | 12 |
| Dòng sau loại duplicate | 7.196 |
| Dòng có giá dương để phát hiện bất thường | 7.193 |
| Dòng dùng huấn luyện regression | 7.141 |
| Giá dương dưới 1 triệu | 49 |
| Giá trên 1 tỷ | 3 |

Các giá dưới 1 triệu hoặc trên 1 tỷ không được dùng để huấn luyện regression vì có thể là lỗi nhập liệu hoặc trường hợp cực đoan. Tuy nhiên, các dòng này vẫn được giữ trong tập anomaly detection.

> **Data usage notice:** Bộ dữ liệu chỉ được sử dụng cho mục đích học tập và nghiên cứu. Không tái phân phối hoặc sử dụng cho mục đích thương mại khi chưa kiểm tra đầy đủ các điều khoản pháp lý và quyền sở hữu dữ liệu.

---

## 4. Data Preparation

### Chuyển đổi dữ liệu

- Chuyển giá từ chuỗi sang đơn vị **triệu đồng**.
- Chuyển năm đăng ký sang dạng số.
- Tạo `age = 2025 - year`.
- Tách quận/huyện từ địa chỉ.
- Tạo độ dài tiêu đề và mô tả.
- Xử lý missing values.
- Loại duplicate theo `Href`.

### Feature được sử dụng

**Feature số**

- `age`;
- `km`;
- `title_length`;
- `description_length`.

**Feature phân loại**

- `brand`;
- `model`;
- `bike_type`;
- `capacity`;
- `origin`;
- `district`;
- `segment`.

### Feature không sử dụng

- `condition`;
- `warranty`;
- `weight`.

Ba cột này gần như chỉ có một giá trị nên không giúp mô hình phân biệt các xe.

Hai cột `Khoảng giá min` và `Khoảng giá max` cũng không được đưa vào mô hình để tránh nguy cơ data leakage. Trong vùng train, chúng có tương quan khoảng 0,768 với giá nhưng chỉ khoảng 28,17% giá thực nằm trong khoảng ước tính có sẵn.

---

## 5. Exploratory Data Analysis

Phân phối giá lệch phải mạnh:

| Thống kê | Giá trị |
|---|---:|
| Median | 16,5 triệu |
| P90 | 65 triệu |
| P99 | 225 triệu |
| Maximum | 136.000 triệu |

Do dữ liệu có các giá cực lớn, target được biến đổi bằng:

```text
log1p(price)
```

trước khi huấn luyện. Prediction được đổi ngược bằng `expm1` trước khi tính metric trên đơn vị triệu đồng.

### Một số thương hiệu phổ biến

| Thương hiệu | Số tin | Giá trung vị |
|---|---:|---:|
| Honda | 4.365 | 19 triệu |
| Yamaha | 1.411 | 13,5 triệu |
| Piaggio | 380 | 25 triệu |
| Suzuki | 280 | 17 triệu |
| Ducati | 16 | 130 triệu |

Sự khác biệt lớn giữa các hãng và dòng xe cho thấy không nên dùng một ngưỡng giá chung cho toàn bộ dữ liệu.

---

## 6. Motorbike Segmentation

Notebook tạo **93 phân khúc**.

| Thống kê phân khúc | Giá trị |
|---|---:|
| Số phân khúc | 93 |
| Kích thước nhỏ nhất | 24 |
| Kích thước trung vị | 51 |
| Kích thước lớn nhất | 255 |

Chiến lược phân khúc:

1. Ưu tiên `brand + model + year_band`.
2. Nếu nhóm quá nhỏ, dùng `brand + bike_type`.
3. Nếu vẫn quá nhỏ, dùng `bike_type + capacity`.
4. Các nhóm hiếm còn lại được gộp thành `Nhóm hiếm`.

Phân khúc được sử dụng:

- như một feature của mô hình regression;
- để tính median, MAD và percentile cho anomaly detection.

Notebook không huấn luyện 93 mô hình riêng vì nhiều phân khúc có số mẫu nhỏ.

---

## 7. Train/Test Design

Dữ liệu 7.141 dòng được chia một lần và dùng chung cho scikit-learn và Spark:

| Tập dữ liệu | Số dòng |
|---|---:|
| Train | 5.712 |
| Test | 1.429 |

Việc chia dữ liệu được stratify theo nhóm giá để train và test có phân bố giá tương đối giống nhau.

Metric chính để chọn mô hình là **MAE**, vì:

- có đơn vị triệu đồng;
- dễ giải thích;
- ít bị một số sai số cực lớn chi phối hơn RMSE.

RMSE và R² vẫn được báo cáo để thể hiện đầy đủ sự đánh đổi giữa các mô hình.

---

# 8. Bài toán 1 — Price Prediction

## 8.1. Ablation Test với Segment

### scikit-learn Random Forest

| Thiết lập | MAE | RMSE | R² |
|---|---:|---:|---:|
| Không segment | 10,206 | 33,717 | 0,450 |
| Có segment | **10,151** | **33,287** | **0,464** |

Segment cải thiện cả ba metric nên được giữ lại.

### Spark Random Forest

| Thiết lập | MAE | RMSE | R² |
|---|---:|---:|---:|
| Không segment | 11,785 | **33,974** | **0,442** |
| Có segment | **11,604** | 34,422 | 0,427 |

Segment cải thiện MAE nhưng làm RMSE và R² giảm nhẹ. Notebook vẫn giữ segment vì MAE là tiêu chí chính và segment còn cần thiết cho anomaly detection.

---

## 8.2. scikit-learn Results

Năm mô hình được thử nghiệm:

1. Linear Regression.
2. Decision Tree.
3. Random Forest.
4. Gradient Boosting.
5. Extra Trees.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Random Forest** | **10,129** | 33,262 | 0,465 |
| Extra Trees | 10,425 | 32,745 | 0,482 |
| Linear Regression | 11,363 | **31,483** | **0,521** |
| Gradient Boosting | 11,796 | 35,558 | 0,389 |
| Decision Tree | 12,099 | 36,550 | 0,354 |

**Model được chọn:** Random Forest, do có MAE thấp nhất.

Linear Regression có RMSE và R² tốt hơn, cho thấy không có một mô hình tốt nhất trên mọi metric. Việc lựa chọn Random Forest dựa trên mục tiêu ưu tiên MAE.

### Sai số theo nhóm giá

| Nhóm giá | Số dòng test | MAE | RMSE |
|---|---:|---:|---:|
| Dưới 10 triệu | 447 | 3,548 | 7,837 |
| 10–30 triệu | 599 | 4,666 | 7,104 |
| 30–100 triệu | 324 | 14,808 | 23,029 |
| Trên 100 triệu | 59 | 89,764 | 151,347 |

Mô hình dự đoán tốt hơn ở nhóm xe phổ thông. Sai số tăng mạnh ở xe trên 100 triệu vì nhóm này có ít mẫu và chứa nhiều xe cao cấp hoặc xe hiếm.

### Feature Importance hàng đầu

| Feature | Importance |
|---|---:|
| Dòng xe SH | 0,239 |
| Tuổi xe | 0,231 |
| Dung tích trên 175 cc | 0,073 |
| Số kilomet | 0,040 |
| Độ dài mô tả | 0,039 |

---

## 8.3. Spark ML Results

Năm mô hình được thử nghiệm:

1. Linear Regression.
2. Generalized Linear Regression.
3. Decision Tree.
4. Random Forest.
5. Gradient-Boosted Trees.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Random Forest** | **10,543** | 33,431 | 0,460 |
| Linear Regression | 11,237 | **30,587** | **0,548** |
| Generalized Linear Regression | 11,237 | **30,587** | **0,548** |
| Gradient-Boosted Trees | 11,474 | 33,248 | 0,466 |
| Decision Tree | 11,862 | 38,708 | 0,276 |

**Model được chọn:** Random Forest, do có MAE thấp nhất.

Generalized Linear Regression được cấu hình với Gaussian family và identity link nên cho kết quả gần như tương đương Linear Regression trong thử nghiệm này.

### Một số phân khúc có sai số Spark cao

| Phân khúc | Số dòng | MAE |
|---|---:|---:|
| Tay côn/Moto — Trên 175 cc | 14 | 117,059 |
| Suzuki — Tay côn/Moto | 31 | 32,440 |
| Suzuki — Xe số | 14 | 30,146 |

### Feature Importance hàng đầu

| Feature | Importance |
|---|---:|
| Dòng xe SH | 0,270 |
| Tuổi xe | 0,248 |
| Dung tích trên 175 cc | 0,087 |
| Loại xe số | 0,039 |
| Số kilomet | 0,023 |

Hai môi trường đều xác định dòng SH, tuổi xe và dung tích trên 175 cc là các feature quan trọng.

> Kết quả scikit-learn và Spark được đánh giá riêng trong từng môi trường. Project không sử dụng các metric này để kết luận môi trường nào tốt hơn.

---

# 9. Bài toán 2 — Price Anomaly Detection

## 9.1. Out-of-Fold Predictions

Residual không được tính từ prediction trên chính dữ liệu model đã học.

Cả hai môi trường sử dụng **5-fold out-of-fold prediction** cho 7.141 dòng training:

- mỗi dòng được dự đoán bởi một model không học từ chính dòng đó;
- các dòng ngoài vùng training được dự đoán bằng final model;
- toàn bộ 7.193 tin có giá dương đều có prediction để tính anomaly.

Kiểm tra Spark OOF:

| Kiểm tra | Kết quả |
|---|---:|
| Expected OOF rows | 7.141 |
| Actual OOF rows | 7.141 |
| ID thiếu hoặc có nhiều prediction | 0 |
| Tổng prediction cuối | 7.193 |

---

## 9.2. Five Anomaly Signals

### Tín hiệu 1 — Residual-Z

```text
residual = actual_price - predicted_price

z = (residual - segment_median)
    / (1.4826 × segment_MAD)
```

Một tin được gắn cờ khi:

```text
|z| >= 3
```

- `z < 0`: giá có xu hướng quá rẻ.
- `z > 0`: giá có xu hướng quá đắt.

### Tín hiệu 2 — P1/P99

Giá bị gắn cờ khi nằm ngoài P1–P99 của phân khúc.

### Tín hiệu 3 — P10/P90

Giá bị gắn cờ khi nằm ngoài dải giá phổ biến P10–P90 của phân khúc.

### Tín hiệu 4 — Unsupervised Detection

- scikit-learn: Isolation Forest.
- Spark: khoảng cách đến tâm KMeans.

Vector đầu vào gồm:

- feature số;
- one-hot feature phân loại;
- giá dự đoán;
- Residual-Z;
- segment.

### Tín hiệu 5 — Composite Score

```text
anomaly_score = 100 × (
    0.40 × residual_signal
  + 0.20 × p01_p99_signal
  + 0.20 × p10_p90_signal
  + 0.20 × unsupervised_signal
)
```

Residual-Z được đặt trọng số cao hơn vì đo trực tiếp chênh lệch giữa giá thực và giá mô hình dự đoán.

Top khoảng 5% anomaly score được đưa vào danh sách ưu tiên kiểm duyệt.

Trọng số và ngưỡng top 5% là lựa chọn heuristic vì dữ liệu chưa có ground truth anomaly để tối ưu trực tiếp.

---

## 9.3. scikit-learn Anomaly Results

| Kết quả | Giá trị |
|---|---:|
| Ngưỡng anomaly score | 49,652 |
| Tổng tin bị gắn cờ | 360 |
| Quá rẻ | 108 |
| Quá đắt | 252 |
| Cờ Residual-Z | 580 |
| Cờ P1–P99 | 236 |
| Cờ P10–P90 | 1.451 |
| Cờ Isolation Forest | 360 |

### Mức đồng thuận giữa bốn tín hiệu thành phần

| Số cờ thành phần | Số tin trong danh sách cuối |
|---:|---:|
| 1 | 9 |
| 2 | 188 |
| 3 | 146 |
| 4 | 17 |

Có **351/360** tin được ít nhất 2/4 tín hiệu thành phần hỗ trợ.

---

## 9.4. Spark Anomaly Results

| Kết quả | Giá trị |
|---|---:|
| Ngưỡng anomaly score | 40,128 |
| Tổng tin bị gắn cờ | 367 |
| Quá rẻ | 108 |
| Quá đắt | 259 |
| Cờ Residual-Z | 581 |
| Cờ P1–P99 | 236 |
| Cờ P10–P90 | 1.451 |
| Cờ KMeans | 367 |

### Mức đồng thuận giữa bốn tín hiệu thành phần

| Số cờ thành phần | Số tin trong danh sách cuối |
|---:|---:|
| 1 | 6 |
| 2 | 204 |
| 3 | 145 |
| 4 | 12 |

Có **361/367** tin được ít nhất 2/4 tín hiệu thành phần hỗ trợ.

Spark có 367 tin thay vì chính xác 5% của 7.193 vì các điểm bằng ngưỡng percentile có thể cùng được chọn.

---

## 9.5. Agreement Between scikit-learn and Spark

| Kết quả | Số tin |
|---|---:|
| Cả hai cùng gắn cờ | 307 |
| Chỉ scikit-learn gắn cờ | 53 |
| Chỉ Spark gắn cờ | 60 |
| Cả hai cùng không gắn cờ | 6.773 |

- **Agreement rate:** 98,43%.
- **Jaccard similarity trên tập anomaly:** 0,731.

Agreement rate cao một phần vì phần lớn tin đều được hai môi trường xem là bình thường. Jaccard phù hợp hơn để đánh giá mức giao nhau của hai danh sách anomaly.

### Case Study

Tin `id = 4239` có:

- giá đăng: **136.000 triệu đồng**, tương đương 136 tỷ đồng;
- dự đoán scikit-learn: khoảng **149,3 triệu đồng**;
- dự đoán Spark: khoảng **143,2 triệu đồng**.

Cả hai môi trường đều xếp đây là một trường hợp cực kỳ bất thường.

---

## 10. Output Files

Sau khi chạy notebook, các file được lưu trong:

```text
/content/project2_outputs/
```

Các output chính:

```text
project2_outputs/
├── motorbikes_cleaned.csv
├── spark_model_data.csv
├── sklearn_segment_ablation.csv
├── spark_segment_ablation.csv
├── sklearn_model_comparison.csv
├── spark_model_comparison.csv
├── sklearn_segment_errors.csv
├── spark_segment_errors.csv
├── sklearn_feature_importance.csv
├── spark_rf_feature_importance.csv
├── sklearn_anomaly_results.csv
├── spark_anomaly_results.csv
├── sklearn_spark_agreement.csv
├── anomaly_case_studies.csv
└── project2_results_summary.json
```

---

## 11. How to Run on Google Colab

### Step 1 — Open the notebook

Upload and open:

```text
project2_motorbike_price_anomaly.ipynb
```

trên Google Colab.

### Step 2 — Prepare the dataset

Đặt file sau trong `/content`:

```text
data_motobikes.xlsx
```

Nếu file chưa tồn tại, notebook sẽ hiển thị hộp thoại upload.

### Step 3 — Run all cells

Trên Colab, chọn:

```text
Runtime → Run all
```

Notebook tự kiểm tra và cài đặt khi cần:

```text
openpyxl
pyspark>=3.5,<4.1
```

### Runtime của lần chạy được báo cáo

```text
Python 3.12.13
Java 17
PySpark 4.0.3
```

Phiên bản thực tế có thể thay đổi theo runtime của Google Colab.

### Step 4 — Download outputs

Các file kết quả nằm trong:

```text
/content/project2_outputs/
```

---

## 12. Suggested Repository Structure

```text
motorbike-price-prediction-anomaly-detection/
├── README.md
├── project2_motorbike_price_anomaly.ipynb
├── data/
│   └── README.md
└── outputs/
    ├── sklearn_model_comparison.csv
    ├── spark_model_comparison.csv
    ├── sklearn_feature_importance.csv
    ├── spark_rf_feature_importance.csv
    ├── sklearn_anomaly_results.csv
    ├── spark_anomaly_results.csv
    ├── sklearn_spark_agreement.csv
    └── anomaly_case_studies.csv
```

Không nên đưa dữ liệu gốc lên repository công khai nếu chưa có quyền tái phân phối.

---

## 13. Limitations

- Dữ liệu không có ground truth anomaly, nên chưa thể tính Precision, Recall hoặc F1 thực sự.
- Một số phân khúc test chỉ có khoảng 10–20 mẫu, khiến metric theo nhóm có thể dao động mạnh.
- Xe hiếm, xe sưu tầm và xe phân khối lớn có sai số cao do ít mẫu và khoảng giá rộng.
- P1/P99 có thể chưa ổn định trong các phân khúc nhỏ.
- KMeans sử dụng `k = 8` như cấu hình thực nghiệm, chưa được tối ưu bằng Silhouette Score.
- Trọng số anomaly và top 5% là heuristic.
- Dữ liệu chỉ là snapshot tại TP.HCM trước ngày 01/07/2025.
- Mô hình cần được theo dõi và huấn luyện lại khi thị trường thay đổi.

---

## 14. Future Improvements

- Thu thập thêm dữ liệu xe cao cấp, xe cổ và xe phân khối lớn.
- Tối ưu hyperparameters cho từng mô hình.
- So sánh thêm XGBoost hoặc LightGBM trong môi trường truyền thống.
- Tối ưu số cụm KMeans bằng Silhouette Score.
- Sử dụng feedback của admin để tạo ground truth anomaly.
- Tối ưu trọng số và threshold bằng dữ liệu đã gắn nhãn.
- Khai thác nội dung tiêu đề và mô tả bằng NLP tiếng Việt.
- Xây dựng giao diện nhập thông tin xe, hiển thị giá gợi ý và từng cờ bất thường.
- Theo dõi data drift và định kỳ retrain mô hình.

---

## 15. Main Conclusions

- Notebook hoàn thành hai bài toán trên cả scikit-learn và Spark ML.
- Random Forest có MAE thấp nhất trong từng môi trường:
  - scikit-learn: **10,129 triệu đồng**;
  - Spark: **10,543 triệu đồng**.
- Mô hình dự đoán tốt hơn ở nhóm xe phổ thông và gặp khó khăn ở xe trên 100 triệu hoặc phân khúc hiếm.
- Anomaly detection triển khai đủ năm bước/tín hiệu và sử dụng out-of-fold predictions.
- scikit-learn cảnh báo 360 tin; Spark cảnh báo 367 tin.
- Có 307 tin được cả hai môi trường cảnh báo.
- Danh sách anomaly có từng cờ, điểm số và lý do để hỗ trợ người kiểm duyệt.
- Hệ thống không thay thế người kiểm duyệt; nó giúp giảm khối lượng kiểm tra thủ công và ưu tiên các tin cần xem trước.

---

## Author Notes

Project được thực hiện cho đồ án tốt nghiệp Data Science & Machine Learning, với mục tiêu học tập và thực hành quy trình xây dựng hệ thống Machine Learning trên cả môi trường truyền thống và Big Data.
