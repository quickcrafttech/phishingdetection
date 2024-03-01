from crypt import methods
from flask import Flask, request, render_template, redirect
from model import Classifier
from model_utils import Model_Utils

app = Flask(__name__)

model = Classifier()


@app.route("/", methods=['GET','POST'])
def index():
    prediction = None
    if request.method == "POST":
        # Get the text from the POST request
        text = request.form["message"]

        # Make prediction using the classifier
        result = model.predict(text)

        if result == 0:
            prediction = "Not Spam"
        elif result == 1:
            prediction = "Spam"

        return render_template("index.html", prediction=prediction)

    if request.method == "GET":
        return render_template("index.html", prediction=None)


if __name__ == "__main__":
    Model_Utils.download_lib()
    app.run(debug=True, host='0.0.0.0')
