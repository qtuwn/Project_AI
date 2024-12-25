# Spam Filter Project

## Mô tả dự án
Dự án này sử dụng thuật toán Naive Bayes để xây dựng một bộ lọc email spam. Dữ liệu được sử dụng là tập dữ liệu SMS Spam Collection, trong đó mỗi mẫu dữ liệu bao gồm một nhãn (0 cho không phải spam, 1 cho spam) và một thông điệp văn bản.

## Cài đặt
Để chạy dự án này, bạn cần cài đặt các thư viện Python sau:

- `nltk`
- `scikit-learn`
- `joblib`
- `pandas`
- `requirements.txt`

Bạn có thể cài đặt các thư viện này bằng cách chạy lệnh sau trong cmd:

```bash
pip install nltk scikit-learn joblib pandas
```

## Hướng dẫn sử dụng
Tải dữ liệu: Đảm bảo rằng bạn có tệp dữ liệu sms_data.csv trong thư mục data.

Chạy script huấn luyện mô hình: Chạy file train_model.py để huấn luyện mô hình và lưu lại mô hình đã huấn luyện.

Dự đoán email spam: Sử dụng mô hình đã huấn luyện để dự đoán xem một email có phải là spam hay không. Chạy file main.py để thực hiện việc này.

## Cấu trúc dự án
data: Thư mục chứa tệp dữ liệu sms_data.csv.
src: Thư mục chứa mã nguồn của dự án.
data_loader.py: Chứa hàm để tải dữ liệu.
preprocessing.py: Chứa hàm tiền xử lý văn bản.
train_model.py: Chứa mã để huấn luyện và lưu mô hình.
email_predictor.py: Chứa mã để tải mô hình và dự đoán.
common.py: Chứa các hàm tiện ích chung.
main.py: Chứa mã để kiểm tra chức năng dự đoán.

## Chi tiết thuật toán
Dự án này sử dụng thuật toán Naive Bayes để phân loại email spam. Các bước chính bao gồm:

Tiền xử lý dữ liệu: Loại bỏ các văn bản rỗng, chuyển đổi văn bản thành dạng số sử dụng TF-IDF Vectorizer.
Chia dữ liệu: Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
Huấn luyện mô hình: Sử dụng thuật toán Naive Bayes để huấn luyện mô hình trên tập huấn luyện.
Đánh giá mô hình: Đánh giá mô hình trên tập kiểm tra và lưu lại mô hình đã huấn luyện.