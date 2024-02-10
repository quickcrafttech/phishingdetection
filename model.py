from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import joblib
from model_utils import Model_Utils

class Classifier:
    def __init__(self):
        # Define file paths relative to the current file
        self.vocabulary_path = 'resources/tfidf_vectorizer_fulldata.joblib'
        self.model_path = 'resources/svm_0.991_fulldata.pkl'

        # Load TF-IDF vectorizer and SVC model
        self.tfidf_vectorizer = joblib.load(self.vocabulary_path)
        self.svm_model = joblib.load(self.model_path)

    def predict(self, text):
        # Preprocess the text using methods from Model_Utils
        preprocessed_text = Model_Utils.remove_punctuation_regex(text)
        preprocessed_text = Model_Utils.remove_stopwords(text)

        # Vectorize the preprocessed text
        text_vectorized = self.tfidf_vectorizer.transform([preprocessed_text])

        # Make predictions using the SVM model
        prediction = self.svm_model.predict(text_vectorized)

        return prediction
