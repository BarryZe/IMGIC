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
from PIL import Image

TEMP_FOLDER = 'static/temp'
UPLOAD_FOLDER = 'static/uploads'
model = ResNet152V2(weights='imagenet')    # Load the pre-trained ResNet50 model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'random string'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER']= TEMP_FOLDER

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
        temp_filename = os.path.join(app.config['TEMP_FOLDER'], file.filename)
        file.save(temp_filename)
        filename=os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        crop_image_to_square(temp_filename,filename )
        results = generate_keywords(filename)
        keywords_string = ', '.join(results)
        print(keywords_string)
        //gen_music(keywords_string)
        return redirect(url_for('index'))
    
def crop_image_to_square(source_path, destination_path):
    img = Image.open(source_path)
    width, height = img.size

    # Find the smaller dimension to crop the square
    min_dim = min(width, height)

    # Calculate coordinates to crop a square
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    # Crop the image to a square
    cropped_img = img.crop((left, top, right, bottom))

    # Save the cropped image back to the file
    cropped_img.save(destination_path)



if __name__ == '__main__':
    app.run(port=8000,debug=True)
    