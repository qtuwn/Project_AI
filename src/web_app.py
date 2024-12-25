from flask import Flask, request, render_template
from email_predictor import load_model, predict
from common import MODEL_PATH, VECTORIZER_PATH, print_log

app = Flask(__name__)

try:
    print_log("Tải mô hình và vectorizer...")
    model = load_model(MODEL_PATH)
    vectorizer = load_model(VECTORIZER_PATH)
    print_log("Tải thành công.", level="SUCCESS")
except Exception as e:
    print_log(f"Lỗi tải mô hình hoặc vectorizer: {e}", level="ERROR")
    model, vectorizer = None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_email():
    try:
        email_text = request.form.get('email_text', '')
        if not email_text:
            return render_template('index.html', prediction="Lỗi: Nội dung email trống", email_text=email_text)

        prediction = predict(model, vectorizer, email_text)
        return render_template('index.html', prediction=prediction, email_text=email_text)
    except Exception as e:
        print_log(f"Lỗi dự đoán: {e}", level="ERROR")
        return render_template('index.html', prediction="Lỗi trong dự đoán", email_text="")

if __name__ == '__main__':
    app.run(debug=True)
