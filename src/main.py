from data_loader import load_data
from train_model import train_and_save_model
try:
    from email_predictor import load_model, predict
except ImportError:
    print("Error: model_loader module not found.")
    exit(1)

# Load data
data_path = 'data/sms_data.csv'  # Đảm bảo rằng đường dẫn này đúng
data = load_data(data_path)

if data is None or data.empty:
    print("[ERROR] Dữ liệu không hợp lệ hoặc rỗng.")
    exit(1)

# Train and save the model
model_path = 'models/spam_model.pkl'
vectorizer_path = 'models/tfidf_vectorizer.pkl'
train_and_save_model(data_path, model_path, vectorizer_path)

# Load the model
model, tfidf = load_model(model_path, vectorizer_path)

# Test the prediction function
test_email = "Congratulations! You've won a free trip to Hawaii. Call now!"
print(predict(model, tfidf, test_email))
