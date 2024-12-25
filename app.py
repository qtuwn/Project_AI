import sys
from flask import Flask, request, render_template
import os
print(os.path.exists('templates/index.html'))  # Nên in ra True nếu tệp tồn tại

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


from email_predictor import load_model, predict
# Khởi tạo Flask và chỉ định đường dẫn đến thư mục templates
app = Flask(__name__, template_folder='src/templates')


# Load the model and vectorizer
model_path = 'models/spam_model.pkl'
vectorizer_path = 'models/tfidf_vectorizer.pkl'
model, vectorizer = load_model(model_path, vectorizer_path)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

@app.route('/predict', methods=['POST'])
def predict_spam():
    email_text = request.form['email_text']  # Lấy nội dung email từ form
    prediction = predict(model, vectorizer, email_text)  # Dự đoán spam hay không
    prediction_text = 'Spam' if prediction == 1 else 'Not Spam'  # Gán kết quả dạng chuỗi
    return render_template('index.html', email_text=email_text, prediction=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)