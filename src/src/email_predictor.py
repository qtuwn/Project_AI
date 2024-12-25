import pickle

def load_model(model_path, vectorizer_path):
    """
    Load the machine learning model and vectorizer from the specified file paths.

    Args:
        model_path (str): Path to the saved model file.
        vectorizer_path (str): Path to the saved vectorizer file.

    Returns:
        tuple: A tuple containing the loaded model and vectorizer.
    """
    # Load the model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Load the vectorizer
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    return model, vectorizer


def predict(model, vectorizer, email_text):
    """
    Predict whether the given email text is spam or not.

    Args:
        model: The machine learning model for classification.
        vectorizer: The TF-IDF vectorizer used for feature extraction.
        email_text (str): The text of the email to classify.

    Returns:
        int: The prediction result (1 for spam, 0 for not spam).
    """
    # Vectorize the email text
    email_vectorized = vectorizer.transform([email_text])
    
    # Make a prediction
    return model.predict(email_vectorized)[0]
