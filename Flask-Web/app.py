# Importing flask - Flask is a micro web framework written in Python.
# It is classified as a microframework because it does not require particular tools or libraries.
import flask as fl


# Importing numpy - Library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.
# Includes a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np

# Importing base64 - Python’s Base64 module provides functions to encode binary data to Base64 encoded format and decode such encodings back to binary data.
import base64

# Imported to recognise images
import cv2

# Library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
from PIL import Image, ImageOps

# Used for importing model into notebook
from keras.models import load_model

# Loading the model
model = load_model('../model.h5')

# Creating the web application
app = fl.Flask(__name__)

# Resizing
height = 28
width = 28
size = height, width

# Routing to index.html - which is my homepage.
@app.route('/')
def home():
    # returns html file (index.html displayed)
    return fl.render_template('index.html')

# Image is sent to this page for prediction
@app.route('/predict', methods=['POST'])
def convertImage():

    encoded = fl.request.values[('imgBase64')]

    decoded = base64.b64decode(encoded[22:])

    # Saving Image
    with open('image.png', 'wb') as f:
        f.write(decoded)

    userImg = Image.open("image.png")

    newImg = ImageOps.fit(userImg, size, Image.ANTIALIAS)
    # Saving new Image
    newImg.save("resizedImg.png")

    # Loading Image
    cv2Image = cv2.imread("resizedImg.png")

    # Converting to greyscale image
    grayScaleImage = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)

    grayScaleArray = np.array(grayScaleImage, dtype=np.float32).reshape(1, 784)
    grayScaleArray /= 255

    # Setters and Getters
    setPrediction = model.predict(grayScaleArray)
    getPrediction = np.array(setPrediction[0])

    predictedNumber = str(np.argmax(getPrediction))
    # Prints predicted number
    print(predictedNumber)

    # The predicted result is returned
    return predictedNumber

# Running Application
app.run(threaded=False)
