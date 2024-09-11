# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('regmodel.pkl')  # Ensure the correct path to your model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # This will render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting form data
    try:
        median_income = float(request.form['median_income'])
        total_rooms = float(request.form['total_rooms'])
        housing_median_age = float(request.form['housing_median_age'])
        total_bedrooms = float(request.form['total_bedrooms'])
        population = float(request.form['population'])
        households = float(request.form['households'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])

        # Prepare the data as a 2D array for the model
        features = np.array([[median_income, total_rooms, housing_median_age,
                              total_bedrooms, population, households, latitude, longitude]])
        
        # Make the prediction
        predicted_price = model.predict(features)[0]

        # Render the result
        return render_template('index.html', prediction_text=f'Predicted House Price: ${predicted_price:,.2f}')

    except ValueError as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
