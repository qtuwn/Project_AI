import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

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
