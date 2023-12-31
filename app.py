import os
from flask import Flask, render_template, request, flash, redirect, url_for
from PIL import Image
from tensorflow import keras
from keras.preprocessing import image
from musicgen import gen_music
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

UPLOAD_FOLDER = 'static/uploads'
TEMP_FOLDER = 'static/temp'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'random string'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER']= TEMP_FOLDER

def generate_alt_text(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    raw_image = Image.open(image_path).convert('RGB')

    # conditional image captioning
    text = "A picture of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

@app.route('/')
def index():
    image_files = os.listdir('static/uploads')
    audio_files = os.listdir('static/output')
    init_files = [pic for pic in image_files if (pic.split('.')[0] + ".wav") in audio_files]
    return render_template('index.html', init_files=init_files)

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
        results = generate_alt_text(filename)
        prompt = "A song about" + results[12:]
        print("Prompt: " + prompt)
        gen_music(prompt, filename)
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
    