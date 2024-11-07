# server.py
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import rotate
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
import os
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model (replace 'model.h5' with your model file)
model = load_model('model.h5')

# Define the class labels
class_labels = ['acne', 'carcinoma', 'eczema', 'keratosis', 'milia', 'rosacea','tick']

def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image to the input size of your model
    img = img.convert('RGB')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the input shape of the model
    img_array /= 255.0  # Normalize the image
    return img_array

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        img = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_image(img)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        return jsonify({'prediction': predicted_class})



x_train = pickle.load(open('train_val_test_sets6/after_aug/x_train', 'rb'))
y_train = pickle.load(open('train_val_test_sets6/after_aug/y_train', 'rb'))
x_val = pickle.load(open('train_val_test_sets6/x_val', 'rb'))
y_val = pickle.load(open('train_val_test_sets6/y_val', 'rb'))
x_test = pickle.load(open('train_val_test_sets6/x_test', 'rb'))
y_test = pickle.load(open('train_val_test_sets6/y_test', 'rb'))

x_t = len(x_train)
x_copy = x_train.copy()
count = 0
ecz, ker, ros = 0, 0, 0
for i in tqdm(range(x_t)):
    if y_train[i] == 0:
        x_train = np.append(x_train, [rotate(x_copy[i], angle=45, mode='wrap')], axis=0)
        y_train = np.append(y_train, 0)

    elif y_train[i] == 2:
        if ecz % 3 == 0:
            x_train = np.append(x_train, [rotate(x_copy[i], angle=45, mode='wrap')], axis=0)
            x_train = np.append(x_train, [np.flipud(x_copy[i])], axis=0)
            for j in range(2):
                y_train = np.append(y_train, 2)
        ecz += 1

    elif y_train[i] == 3:
        if ker % 6 == 0:
            x_train = np.append(x_train, [rotate(x_copy[i], angle=45, mode='wrap')], axis=0)
            x_train = np.append(x_train, [np.flipud(x_copy[i])], axis=0)
            for j in range(2):
                y_train = np.append(y_train, 3)
        ker += 1

    elif y_train[i] == 4:
        x_train = np.append(x_train, [rotate(x_copy[i], angle=45, mode='wrap')], axis=0)
        x_train = np.append(x_train, [np.flipud(x_copy[i])], axis=0)
        x_train = np.append(x_train, [np.fliplr(x_copy[i])], axis=0)
        x_train = np.append(x_train, [rotate(np.fliplr(x_copy[i]), angle=270, mode='wrap')], axis=0)
        for j in range(4):
            y_train = np.append(y_train, 4)

    elif y_train[i] == 5:
        if ros % 9 == 0:
            x_train = np.append(x_train, [rotate(x_copy[i], angle=45, mode='wrap')], axis=0)
            x_train = np.append(x_train, [np.flipud(x_copy[i])], axis=0)
            for j in range(2):
                y_train = np.append(y_train, 5)
        ros += 1

indices = np.random.permutation(len(x_train))
x_train = x_train[indices]
y_train = y_train[indices]

f = open("dir../after_aug/x_train", "wb")
pickle.dump(x_train, f)
f.close()
f = open("dir../after_aug/y_train", "wb")
pickle.dump(y_train, f)
f.close()

f = open("dir../after_aug/x_val", "wb")
pickle.dump(x_val, f)
f.close()
f = open("dir../after_aug/y_val", "wb")
pickle.dump(y_val, f)
f.close()

f = open("dir../after_aug/x_test", "wb")
pickle.dump(x_test, f)
f.close()
f = open("dir../after_aug/y_test", "wb")
pickle.dump(y_test, f)
f.close()

