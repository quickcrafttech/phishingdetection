# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from model_utils import Model_Utils

class Classifier:
    """Pre-trained spam classification model using TF-IDF and SVM.

    This class loads a pre-trained Support Vector Machine (SVM) model that
    classifies text as spam or not spam. It utilizes a TF-IDF vectorizer to
    convert text into numerical features suitable for machine learning algorithms.

    **Note:** This class assumes the pre-trained model and vectorizer files
    (specified in `self.vocabulary_path` and `self.model_path`) already exist.
    The training process for these models is not included here.
    """

    def __init__(self):
        """Loads the pre-trained TF-IDF vectorizer and SVM model."""

        # Define file paths relative to the current file for the saved models
        self.vocabulary_path = 'resources/tfidf_vectorizer_fulldata.joblib'
        self.model_path = 'resources/svm_0.991_fulldata.pkl'

        # Load the pre-trained TF-IDF vectorizer
        self.tfidf_vectorizer = joblib.load(self.vocabulary_path)
        # Load the pre-trained SVM model
        self.svm_model = joblib.load(self.model_path)

    def predict(self, text):
        """Predicts whether a given text string is spam or not spam.

        This method performs the following steps:

        1. Preprocesses the text using `Model_Utils.remove_punctuation_regex`
           and `Model_Utils.remove_stopwords` to remove punctuation and stop words.
        2. Converts the preprocessed text into a numerical representation
           using the loaded TF-IDF vectorizer.
        3. Makes a prediction on the transformed text using the loaded SVM model.

        Args:
            text (str): The text string to classify as spam or not spam.

        Returns:
            int: The predicted class label (0: Not Spam, 1: Spam).
        """

        # Preprocess the text using methods from Model_Utils for cleaning
        preprocessed_text = Model_Utils.remove_punctuation_regex(text)
        preprocessed_text = Model_Utils.remove_stopwords(text)

        # Convert the preprocessed text into a TF-IDF vector
        text_vectorized = self.tfidf_vectorizer.transform([preprocessed_text])

        # Make a prediction using the pre-trained SVM model
        prediction = self.svm_model.predict(text_vectorized)

        return prediction  # Return the first element (predicted class)
