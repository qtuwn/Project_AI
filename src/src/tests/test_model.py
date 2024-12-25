import pytest
from email_predictor import load_model, predict

@pytest.fixture
def setup():
    model, vectorizer = load_model('src/spam_classifier.pkl')
    return model, vectorizer

def test_spam(setup):
    model, vectorizer = setup
    email = "Win $1000 now by clicking this link!"
    assert predict(model, vectorizer, email) == "spam"

def test_ham(setup):
    model, vectorizer = setup
    email = "Hi, can we schedule a meeting for tomorrow?"
    assert predict(model, vectorizer, email) == "ham"
