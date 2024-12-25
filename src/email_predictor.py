import joblib
from preprocessing import preprocess_text
from common import MODEL_PATH, VECTORIZER_PATH

def load_model(model_path, vectorizer_path):
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None, None

def predict(model, vectorizer, text):
    """
    Dự đoán văn bản là spam hoặc ham.
    """
    try:
        text = preprocess_text(text)
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)
        return 'ham' if prediction == 0 else 'spam'
    except Exception as e:
        print(f"Lỗi dự đoán: {e}")
        return "Error"

if __name__ == "__main__":
    model, vectorizer = load_model(MODEL_PATH, VECTORIZER_PATH)

    if model is not None and vectorizer is not None:
        # Ví dụ về cách sử dụng hàm predict
        email_text = "Subject: Important: Verify Your Bank Account Dear Customer, Your account has been locked due to suspicious activity. Please click the link below to verify your account immediately: [Fake Link] Failure to do so may result in account deactivation."
        prediction = predict(model, vectorizer, email_text)
        print(f"Dự đoán: {prediction}")
    else:
        print("Không thể tải mô hình và vectorizer.")
