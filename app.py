import sys
import os
from flask import Flask, request, render_template
import joblib

# Thêm src vào sys.path để import module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train_model import preprocess_text

# Cập nhật đường dẫn đến thư mục templates
app = Flask(__name__, template_folder='src/templates')

# Load model và vectorizer
model_path = 'models/spam_filter_model.pkl'
vectorizer_path = 'models/vectorizer.pkl'
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict(model, vectorizer, email_text):
    email_text = preprocess_text(email_text)
    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_spam():
    email_text = request.form['email_text']  # Lấy nội dung email từ form
    prediction = predict(model, vectorizer, email_text)  # Dự đoán spam hay không
    prediction_text = 'Spam' if prediction == 'spam' else 'Not Spam'  # Gán kết quả dạng chuỗi
    return render_template('index.html', email_text=email_text, prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
