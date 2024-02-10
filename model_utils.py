import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Model_Utils:
    @staticmethod
    def remove_punctuation_regex(text):
        # Define the regex pattern to match punctuation
        pattern = r'[^\w\s]'  # Matches any character that is not a word character or whitespace
        # Use the sub() function to replace punctuation with an empty string
        text_no_punct = re.sub(pattern, '', text)
        return text_no_punct

    @staticmethod
    def remove_stopwords(text):
        # Tokenize the text
        tokens = word_tokenize(text.lower())  # Convert to lowercase
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Join the tokens back into a string
        clean_text = ' '.join(filtered_tokens)
        return clean_text