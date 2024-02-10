from flask import Flask, request, render_template, redirect
from model import Classifier

app = Flask(__name__)

model = Classifier()

@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get the text from the POST request
        text = request.form['message']

        # Make prediction using the classifier
        result = model.predict(text)

        if result==0:
            prediction = 'Not Spam'
        elif result==1:
            prediction = 'Spam'
        
        return render_template('index.html', prediction=prediction)

    if request.method == 'GET':
        return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
    