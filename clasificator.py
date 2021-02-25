import math
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ======================== FRAMES (TRAINING AND TEST)
count = 0
videoFile = "training/TomJerry.mp4"
cap = cv2.VideoCapture(videoFile)  # capturing the video from the given path
frameRate = cap.get(5)  # frame rate
x = 1

while cap.isOpened():
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    if ret is not True:
        break
    if frameId % math.floor(frameRate) == 0:
        filename = "frames/frame%d.jpg" % count
        count += 1
        cv2.imwrite(filename, frame)
cap.release()
print("Done!")

count = 0
videoFile = "test/TomJerryTest.mp4"
cap = cv2.VideoCapture(videoFile)  # capturing the video from the given path
frameRate = cap.get(5)  # frame rate
x = 1

while cap.isOpened():
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    if ret is not True:
        break
    if frameId % math.floor(frameRate) == 0:
        filename = "frames/test%d.jpg" % count
        count += 1
        cv2.imwrite(filename, frame)
cap.release()
print("Done!")

# ======================== TRAINING

# Load data
data = pd.read_csv('training/mapping.csv')

# Array de imagenes -> Cada img es una matriz de pixeles (R,G,B)
X = []
for img_name in data.Image_ID:
    img = plt.imread('frames/' + img_name)
    X.append(img)
X = np.array(X)

# one hot encoding Clases
y = data.Class
train_y = np_utils.to_categorical(y)

# Redimensionamos las imagenes a 224 x 224 x 3
image = []
for i in range(0, X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224, 224, 3)).astype(int)
    image.append(a)
X = np.array(image)

# Preprocessing input data -> Mejora el rendimiento
X = preprocess_input(X)

# Dividir aleatoriamente imgs en entrenamiento y validacion
X_train, X_valid, y_train, y_valid = train_test_split(X, train_y, test_size=0.3, random_state=42)

# Construccion del modelo -> usa VGG16 pretrained model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)

X_train = X_train.reshape(208, 7 * 7 * 512)
X_valid = X_valid.reshape(90, 7 * 7 * 512)

train = X_train / X_train.max()
X_valid = X_valid / X_train.max()

model = Sequential()
model.add(InputLayer((7 * 7 * 512,)))  # input layer
model.add(Dense(units=500, activation='relu'))  # hidden layer
model.add(Dropout(0.5))  # adding dropout
model.add(Dense(units=300, activation='relu'))  # hidden layer
model.add(Dropout(0.5))  # adding dropout
model.add(Dense(units=100, activation='relu'))  # hidden layer
model.add(Dropout(0.5))  # adding dropout
model.add(Dense(3, activation='softmax'))  # output layer

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))

# ======================== TEST
test = pd.read_csv('test/testing.csv')

test_image = []
for img_name in test.Image_ID:
    img = plt.imread('frames/' + img_name)
    test_image.append(img)
test_img = np.array(test_image)

test_y = np_utils.to_categorical(test.Class)

test_image = []
for i in range(0, test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224, 224)).astype(int)
    test_image.append(a)

test_image = np.array(test_image)
test_image = preprocess_input(test_image)
test_image = base_model.predict(test_image)
test_image = test_image.reshape(186, 7 * 7 * 512)
test_image = test_image / test_image.max()

# ======================== SCORES

print("======================================================")
scores = model.evaluate(test_image, test_y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print("======================================================")
predictions = np.argmax(model.predict(test_image), axis=-1)
print("The screen time of JERRY is", predictions[predictions == 1].shape[0], "seconds")
print("The screen time of TOM is", predictions[predictions == 2].shape[0], "seconds")
