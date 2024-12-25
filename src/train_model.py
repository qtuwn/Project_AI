import os
import joblib
import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Thêm thư mục src vào sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_loader import load_data
from common import create_directory_if_not_exists, print_log

# Thiết lập mã hóa UTF-8 cho đầu ra
sys.stdout.reconfigure(encoding='utf-8')

def preprocess_text(text):
    # Chuyển đổi văn bản thành chữ thường
    text = text.lower()
    # Loại bỏ các ký tự đặc biệt và số
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # Loại bỏ từ dừng
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def train_and_save_model(data_path, model_path, vectorizer_path):
    try:
        print_log("Tải dữ liệu...")
        data = load_data(data_path)
        if data is None or data.empty:
            print_log("Dữ liệu không hợp lệ hoặc rỗng.", level="ERROR")
            return

        print_log("Tiền xử lý dữ liệu...")
        data['message'] = data['message'].apply(preprocess_text)  # Sử dụng tên cột đúng

        print_log("Chia dữ liệu...")
        X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

        print_log("Huấn luyện mô hình...")
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        # Sử dụng Multinomial Naive Bayes
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        print_log("Đánh giá mô hình...")
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tfidf)
        print(f"Độ chính xác: {accuracy_score(y_test, y_pred):.4f}")
        print("Báo cáo phân loại:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        # ROC-AUC
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()

        print_log("Lưu mô hình...")
        create_directory_if_not_exists(os.path.dirname(model_path))
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print_log("Huấn luyện và lưu thành công.", level="SUCCESS")
    except Exception as e:
        print_log(f"Lỗi trong huấn luyện mô hình: {e}", level="ERROR")

if __name__ == "__main__":
    DATA_PATH = "C:/Users/tuquo/Desktop/Hoc_Tap/project_AI/spam_filter_project/data/sms_data.csv"
    MODEL_PATH = "models/spam_filter_model.pkl"
    VECTORIZER_PATH = "models/vectorizer.pkl"
    train_and_save_model(DATA_PATH, MODEL_PATH, VECTORIZER_PATH)
