# Import necessary libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import ssl


class Model_Utils:
    """Helper functions for text cleaning and data preparation."""

    @staticmethod
    def download_lib():
        """Downloads necessary libraries from NLTK for text processing.

        This function downloads the 'stopwords' and 'punkt' datasets from NLTK,
        which are required for removing stop words and tokenizing text, respectively.

        NLTK (Natural Language Toolkit) is a popular library for natural language
        processing tasks. Downloading these datasets might require an internet connection.

        **Note:** This function handles potential issues with insecure HTTPS contexts
        (advanced topic). You can usually ignore this unless you encounter specific errors.
        """

        try:
            # Handle potential issue with insecure HTTPS contexts (advanced)
            _create_unverified_https_context = ssl._create_unverified_https_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download the stopwords list (common words like "the", "a")
        nltk.download("stopwords")
        # Download the punkt tokenizer for splitting text into words
        nltk.download("punkt")

    @staticmethod
    def remove_punctuation_regex(text):
        """Removes punctuation characters from a given text string.

        This function uses a regular expression to identify and remove all characters
        that are not words or whitespace (spaces, tabs, newlines). This helps focus the
        analysis on the meaningful content of the text.

        Args:
            text (str): The text string from which to remove punctuation.

        Returns:
            str: The text string with punctuation characters removed.
        """

        # Define a regular expression pattern to match punctuation characters
        pattern = r"[^\w\s]"  # Matches anything that's not a word character or whitespace
        # Use the sub() function to replace punctuation with an empty string
        text_no_punct = re.sub(pattern, "", text)
        return text_no_punct

    @staticmethod
    def remove_stopwords(text):
        """Removes common stop words (e.g., "the", "a") from a given text string.

        This function helps focus the analysis on the most important words in the text
        by removing frequently occurring words that don't carry much meaning.

        Args:
            text (str): The text string from which to remove stop words.

        Returns:
            str: The text string with stop words removed.
        """

        # Convert the text to lowercase for case-insensitive stop word removal
        text = text.lower()

        # Split the text into individual words (tokens)
        tokens = word_tokenize(text)

        # Get a list of English stop words from NLTK
        stop_words = set(stopwords.words("english"))

        # Remove stop words from the list of tokens
        filtered_tokens = [word for word in tokens if word not in stop_words]

        # Join the remaining tokens back into a single string
        clean_text = " ".join(filtered_tokens)
        return clean_text
