from tensorflow.keras.models import load_model

import os
from flask import Flask, render_template, request
import pickle
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

app = Flask(__name__)

# # Load the trained model
# model_path = './models/'
# model_file = os.path.join(model_path, 'model.pkl')
# with open(model_file, 'rb') as f:
#     model = pickle.load(f)

# Load the trained model
model_path = './models/'
model_file = os.path.join(model_path, 'model.h5')  
model = load_model(model_file)

# Image preprocessing function
def preprocess_image(image_path, img_size=(256, 256)):
    image = Image.open(image_path)
    image = image.resize(img_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded'

    image_file = request.files['image']
    image_path = f'static/uploads/{image_file.filename}'
    image_file.save(image_path)

    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    prediction_label = 'Glaucoma' if prediction[0][0] > 0.5 else 'Normal'

    return render_template('result.html', prediction=prediction_label, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)