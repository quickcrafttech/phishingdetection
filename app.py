# Import necessary libraries
from flask import Flask, request, render_template, redirect  # Web framework for building web applications
from model import Classifier  # Your custom class containing the spam classification model
from model_utils import Model_Utils  # Helper functions for model-related tasks (if applicable)

# Create a Flask application instance
app = Flask(__name__)

# Load the spam classification model (assuming it's already trained)
model = Classifier()

# Define the main route ("/") handling both GET and POST requests
@app.route("/", methods=['GET', 'POST'])
def index():
    """Processes user input for spam classification and displays the prediction.

    Returns:
        The rendered "index.html" template with the predicted spam classification.
    """

    prediction = None  # Initialize prediction variable for storing the result

    if request.method == "POST":
        """Handles POST requests containing user input for spam classification."""

        # Extract the text from the user's form submission
        text = request.form["message"]

        # Make a prediction using the loaded model
        result = model.predict(text)

        # Convert the numeric result to a human-readable prediction
        if result == 0:
            prediction = "Not Spam"
        elif result == 1:
            prediction = "Spam"

        # Render the "index.html" template with the predicted label
        return render_template("index.html", prediction=prediction)

    if request.method == "GET":
        """Handles GET requests, displaying the initial form."""

        # Render the "index.html" template without a prediction (initial state)
        return render_template("index.html", prediction=None)

# Run the Flask application when executed directly
if __name__ == "__main__":
    # Perform any model-related setup tasks using Model_Utils (if applicable)
    # Example: Downloading required libraries or loading pre-trained models
    Model_Utils.download_lib()

    # Start the Flask application in debug mode, accessible by any device on the network
    app.run(debug=True, host='0.0.0.0')
