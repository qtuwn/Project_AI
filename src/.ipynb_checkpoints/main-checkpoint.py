import pickle
import numpy as np
from data_loader import load_data
from train_model import train_and_save_model

# Load data
data = load_data('SMSSpamCollection')

# Train and save the model
train_and_save_model(data, 'spam_model.pkl')

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model_data = pickle.load(file)
    return model_data['model'], model_data['tfidf']

def predict_email(email_text, model, tfidf):
    email_tfidf = tfidf.transform([email_text])
    prediction = model.predict(email_tfidf)
    return 'spam' if prediction[0] == 1 else 'ham'

# Load the model
model, tfidf = load_model('spam_model.pkl')

# Test the prediction function
test_email = "Congratulations! You've won a free trip to Hawaii. Call now!"
print(predict_email(test_email, model, tfidf))
