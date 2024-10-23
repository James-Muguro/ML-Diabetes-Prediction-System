# Importing essential libraries
from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
import numpy as np

# Load the Random Forest Classifier model
MODEL_FILENAME = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(MODEL_FILENAME, 'rb'))

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management (flash messages)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collecting and validating input data
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])

            # Basic validation
            if any(x < 0 for x in [preg, glucose, bp, st, insulin, age]) or bmi < 0 or dpf < 0:
                flash("Please enter valid non-negative values for all fields.")
                return redirect(url_for('home'))

            # Preparing data for prediction
            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = classifier.predict(data)

            return render_template('result.html', prediction=my_prediction[0])  # Assuming model returns a single value

        except ValueError as e:
            flash("Invalid input. Please ensure all fields are filled out correctly.")
            return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)