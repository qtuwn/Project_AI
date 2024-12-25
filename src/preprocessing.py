import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Tải các tài nguyên cần thiết (chạy một lần)
nltk.download('punkt')       # Dùng để tokenize
nltk.download('wordnet')     # Dùng để lemmatize
nltk.download('stopwords')   # Dùng để loại bỏ stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Tiền xử lý văn bản: loại bỏ stopwords, viết thường, lemmatization.
    """
    try:
        # Chuyển văn bản thành chữ thường
        text = text.lower()
        # Tokenize văn bản
        words = word_tokenize(text)
        # Loại bỏ stopwords và lemmatize
        processed_words = [
            lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words
        ]
        return ' '.join(processed_words)
    except Exception as e:
        print(f"Lỗi tiền xử lý văn bản: {e}")
        return ""