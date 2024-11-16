import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights(r'model_weights\vgg_unfrozen.weights.h5')
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((240, 240))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model_03.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return result01

@app.route('/', methods=['GET'])
def index():
    html_content = '''
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brain Tumor Classification</title>
    </head>
    <body>
        <nav class="navbar navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="#">Brain Tumor Classification</a>
            </div>
        </nav>

        <h2>Welcome to the Brain Tumor Classification App!</h2>
        <p>Upload an image for classification:</p>

        <form id="upload-file" method="post" action="/predict" enctype="multipart/form-data">
            <input type="file" name="file" class="btn btn-success" id="imageUpload" accept=".png, .jpg, .jpeg">
            <button type="submit" class="btn btn-primary">Submit</button>
        </form>

        <h3 id="result"></h3>
    </body>
    </html>
    '''
    return render_template_string(html_content)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Create uploads directory if it doesn't exist
        upload_folder = os.path.join(os.path.dirname(__file__), 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value) 
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
