from PIL import Image
from keras.preprocessing import image
import os
import numpy as np
import pandas as pd
from subprocess import check_output
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib

trainpath = "/content/gdrive/My Drive/diabeticdata/train/"
labelpath = "/content/gdrive/My Drive/diabeticdata/trainLabels.csv"
trainLabels = pd.read_csv(labelpath)
print(trainLabels.head())
listing = os.listdir(trainpath)
print(np.size(listing))

img_rows, img_cols = 224, 224
immatrix = []
imlabel = []

for file in listing:
    base = os.path.basename(trainpath + file)
    fileName = os.path.splitext(base)[0]
    imlabel.append(trainLabels.loc[trainLabels.image==fileName, 'level'].values[0])
    im = Image.open(trainpath + file)
    img = np.array(im.resize((img_rows,img_cols)))
    immatrix.append(np.array(img))

import random

# define transformation methods
def horizontal_flip(image_array):
    return image_array[:, ::-1]

def vertical_flip(image_array):
    return image_array[::-1,:]

def random_transform(image_array):
    if random.random() < 0.5:
        return vertical_flip(image_array)
    else:
        return horizontal_flip(image_array)

from sklearn.utils import shuffle

data,label = shuffle(immatrix, imlabel, random_state=42)
train_data = [data,label]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_data[0], train_data[1], test_size = 0.1, random_state = 42)

print(np.array(x_train).shape)
print(np.array(y_train).shape)

from keras.utils import np_utils

y_train = np_utils.to_categorical(np.array(y_train), 5)
y_test = np_utils.to_categorical(np.array(y_test), 5)

x_train = np.array(x_train).astype("float32")/255.
x_test = np.array(x_test).astype("float32")/255.
print(np.array(x_test).shape)

print(np.array(y_train).shape)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def createModel(LR, EPOCHS, NUM_CLASSES, inputShape):
    model = Sequential()
    # first set of CONV => RELU => MAX POOL layers
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputShape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(activation='softmax', units=NUM_CLASSES))
    # returns our fully constructed deep learning + Keras image classifier
    opt = Adam(lr=LR, decay=LR / EPOCHS)
    # use binary_crossentropy if there are two classes
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

lr = 0.0001
epochs = 10
classnum = 5
model = createModel(lr, epochs, classnum, x_train[0].shape)
model.summary()

model.fit(x_train, y_train, batch_size = 64, epochs=10, shuffle=True, verbose=2)

predictions = model.predict(x_test)
predictions

score = model.evaluate(x_train, y_train, verbose=0)
print(score)