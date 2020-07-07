from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'spam-classifier-multiNB-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open("cv-transform.pkl",'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    if request.method == "POST":
        message = request.form['message']
        data = [message]
        message=cv.transform(data).toarray()
        predictions = classifier.predict(message)
        return render_template("result.html",prediction=predictions)
if __name__ == '__main__':
	app.run(debug=True)