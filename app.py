import cv2
import os
from flask import Flask, render_template, request, flash, redirect, url_for
import tensorflow as tf
import tensorflow

from tensorflow import keras
from keras.applications.resnet_v2 import ResNet152V2, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
from musicgen import gen_music


UPLOAD_FOLDER = 'static/uploads'
model = ResNet152V2(weights='imagenet')    # Load the pre-trained ResNet50 model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def generate_keywords(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get model predictions
    predictions = model.predict(img_array)

    # Decode predictions into human-readable labels
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Extract keywords from predictions
    keywords = [label.replace('_', ' ') for (_, label, _) in decoded_predictions]

    return keywords

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print(request.files)
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        results = generate_keywords(filename)
        keywords_string = ', '.join(results)
        print(keywords_string)
        gen_music(keywords_string)
        return redirect(url_for('index'))
        return str(results)


if __name__ == '__main__':
    app.run(port=8000,debug=True)
    