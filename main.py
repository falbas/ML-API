from tensorflow import keras
import matplotlib.image as mpimg
import cv2 as cv 

def img_classification(filename):
    model = keras.models.load_model('my_model')

    gambar = mpimg.imread(filename)

    gambar = cv.resize(gambar, (28,28), interpolation = cv.INTER_AREA)

    gray = cv.cvtColor(gambar, cv.COLOR_BGR2GRAY)
    gray = 255-gray
    gray = gray/255.0

    label = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

    gray = gray.reshape(1, gray.shape[0], gray.shape[1], 1)

    pred = model.predict(gray)

    return label[pred.argmax()]

import time
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    curr_time = round(time.time()*1000)
    filename = str(curr_time)+file.filename
    file.save(filename)

    pred = img_classification(filename)
    return pred

if __name__ == '__main__':
    app.run(debug=True)