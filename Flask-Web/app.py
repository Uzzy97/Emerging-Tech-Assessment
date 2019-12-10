# Importing flask
import flask as fl


# Importing numpy
import numpy as np

# Used for encoding and decoding data
import base64


import cv2


from PIL import Image, ImageOps

# Used for importing model into notebook
from keras.models import load_model


model = load_model('../model.h5')

app = fl.Flask(__name__)


height = 28
width = 28
size = height, width

# Routing to index.html
@app.route('/')
def home():
    # returns html file
    return fl.render_template('index.html')


@app.route('/predict', methods=['POST'])
def convertImage():
    
    encoded = fl.request.values[('imgBase64')]

    
    decoded = base64.b64decode(encoded[22:])

    
    with open('image.png', 'wb') as f:
        f.write(decoded)

    
    userImage = Image.open("image.png")

    
    newImage = ImageOps.fit(userImage, size, Image.ANTIALIAS)

    
    newImage.save("resizedImg.png")

    
    cv2Image = cv2.imread("resizedImg.png")

    
    grayScaleImage = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
   
    grayScaleArray = np.array(grayScaleImage, dtype=np.float32).reshape(1, 784)
    grayScaleArray /= 255

    
    setPrediction = model.predict(grayScaleArray)
    getPrediction = np.array(setPrediction[0])

    
    predictedNumber = str(np.argmax(getPrediction))
    print(predictedNumber)

    
    return predictedNumber


app.run(threaded=False)
