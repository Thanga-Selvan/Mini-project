from cv2 import cv2 
import numpy as np
from PIL import Image
import base64
import re
from io import BytesIO
import keras

class_name = {0: 'ञ', 1: 'ट', 2: 'ठ', 3: 'ड', 4: 'ढ', 5: 'ण',
              6: 'त', 7: 'थ', 8: 'द', 9: 'ध', 10: 'क', 11: 'न',
              12: 'प', 13: 'फ', 14: 'ब', 15: 'भ',
              16: 'म', 17: 'य', 18: 'र', 19: 'ल',
              20: 'व', 21: 'ख', 22: 'श', 23: 'ष',
              24: 'स', 25: 'ह', 26: 'क्ष', 27: 'त्र',
              28: 'ज्ञ', 29: 'ग', 30: 'घ', 31: 'ङ',
              32: 'च', 33: 'छ', 4: 'ज', 35: 'झ',
              36: '०', 37: '१', 38: '२', 39: '३',
              40: '४', 41: '५', 42: '६', 43: '७',
              44: '८', 45: '९'}


def crop_image(rgb, gray, tol=0):
    mask = gray > tol
    np_ix = np.ix_(mask.any(1), mask.any(0))

    row_start = np_ix[0][0][0]
    row_end = np_ix[0][-1][0]
    column_start = np_ix[1][0][0]
    column_end = np_ix[1][0][-1]
    return rgb[row_start: row_end, column_start: column_end, :]


def prediction(image):
    model = keras.models.load_model('./four_stacked_model.h5')

    rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    rgb_image = cv2.bitwise_not(rgb_image)
    rgb_image = rgb_image / 255.0

    gray_image = cv2.bitwise_not(gray_image)
    gray_image = gray_image / 255.0

    cropped_img = crop_image(rgb_image, gray_image)
    cropped_img = cv2.resize(cropped_img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)

    img_predict = model.predict(cropped_img.reshape((1, 128, 128, 3)))

    return class_name[np.argmax(img_predict)]


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    image = request.values['imageBase64']
    image = re.sub('^data:image/.+;base64,', '', image)
    image = base64.b64decode(image)
    image = Image.open(BytesIO(image))
    label = prediction(image)
    return jsonify('', render_template('predict.html', predClass=label))


if __name__ == "__main__":
    app.run()
