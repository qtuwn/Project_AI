import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_text
from data_loader import load_data
from common import DATA_PATH, MODEL_PATH, VECTORIZER_PATH, create_directory_if_not_exists, print_log
import sys
import io
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def preprocess_data(data):
    """
    Tiền xử lý dữ liệu và loại bỏ các văn bản rỗng.
    """
    data['message'] = data['message'].apply(preprocess_text)
    # Loại bỏ các hàng bị trống sau tiền xử lý
    data = data[data['message'].str.strip() != '']
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['message'])
    y = data['label']
    return X, y, vectorizer


def train_and_save_model(data_path, model_path, vectorizer_path):
    try:
        print_log("Tải dữ liệu...")
        data = load_data(data_path)
        if data is None or data.empty:
            raise ValueError("Không tải được dữ liệu hoặc dữ liệu rỗng.")

        print_log("Tiền xử lý dữ liệu...")
        X, y, vectorizer = preprocess_data(data)

        print_log("Chia dữ liệu...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print_log("Huấn luyện mô hình...")
        model = MultinomialNB()
        model.fit(X_train, y_train)

        print_log("Đánh giá mô hình...")
        y_pred = model.predict(X_test)
        print(f"Độ chính xác: {accuracy_score(y_test, y_pred):.4f}")
        print("Báo cáo phân loại:")
        print(classification_report(y_test, y_pred))

        print_log("Lưu mô hình...")
        create_directory_if_not_exists(os.path.dirname(model_path))
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print_log("Huấn luyện và lưu thành công.", level="SUCCESS")
    except Exception as e:
        print_log(f"Lỗi trong huấn luyện mô hình: {e}", level="ERROR")

if __name__ == "__main__":
    train_and_save_model(DATA_PATH, MODEL_PATH, VECTORIZER_PATH)

from data_loader import load_data

data_path = "C:/Users/tuquo/Desktop/Hoc_Tap/spam_filter_project/data/sms_data.csv"  # Đường dẫn tới tệp dữ liệu
data = load_data(data_path)

if data is None or data.empty:
    print("[ERROR] Dữ liệu không hợp lệ hoặc rỗng.")
else:
    print("[INFO] Dữ liệu đã được tải thành công.")
    print(data.head())  # In 5 dòng đầu tiên để kiểm tra
