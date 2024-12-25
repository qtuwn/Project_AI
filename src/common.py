import os

# Đường dẫn cơ bản cho các tệp dữ liệu và mô hình
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'sms_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'src', 'spam_classifier.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'src', 'spam_vectorizer.pkl')

def create_directory_if_not_exists(path):
    """
    Tạo thư mục nếu chưa tồn tại.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def print_log(message, level="INFO"):
    """
    Ghi log đơn giản.
    """
    print(f"[{level}] {message}")
