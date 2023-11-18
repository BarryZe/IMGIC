import cv2
import tensorflow as tf
import tensorflow

from tensorflow import keras
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
image_path = 'cat picture.jpg'

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

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

# Replace 'your_image.jpg' with the path to your image file
image_path = 'cat picture.jpg'
result_keywords = generate_keywords(image_path)

print("Keywords for the image:", result_keywords)
