import streamlit as st
from email_predictor import load_model, predict

# Tải mô hình và vectorizer
model, vectorizer = load_model('src/spam_classifier.pkl')

# Giao diện
st.title("Spam Filter App")
email_text = st.text_area("Enter Email:")
if st.button("Predict"):
    prediction = predict(model, vectorizer, email_text)
    st.write(f"Prediction: {prediction}")
