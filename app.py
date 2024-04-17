import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the train and test data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Load the trained model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    data = request.form.to_dict()

    # Preprocess input data
    input_data = [data['age'], data['gender'], len(data['chief_complaint']), ...]
    input_data = scaler.transform([input_data])

    # Make prediction
    prediction = model.predict(input_data)

    # Return prediction to the user
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)