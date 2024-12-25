import joblib
from preprocessing import preprocess_text
from common import MODEL_PATH, VECTORIZER_PATH

def load_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

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
        email_text = input("Nhập tin nhắn để dự đoán: ")
        result = predict(model, vectorizer, email_text)
        print(f"Tin nhắn được phân loại là: {result}")
    else:
        print("Không thể tải mô hình hoặc vectorizer.")

