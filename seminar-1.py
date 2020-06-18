
from PIL import Image
from keras.preprocessing import image
import os
import numpy as np
import pandas as pd
from subprocess import check_output
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib

trainpath = "/home/greenjoa/data/diabetic/train/"
labelpath = "/home/greenjoa/data/diabetic/trainLabels.csv"

trainLabels = pd.read_csv(labelpath)
print(trainLabels.head())
listing = os.listdir(trainpath)
print(np.size(listing))
trainLabels['level'].hist(figsize=(10,5))

img_rows, img_cols = 224, 224
immatrix = []
imlabel = []

for file in listing:
    base = os.path.basename(trainpath + file)  
    fileName = os.path.splitext(base)[0]
    imlabel.append(trainLabels.loc[trainLabels.image==fileName, 'level'].values[0])
    im = Image.open(trainpath + file)
    img = im.resize((img_rows,img_cols))
    img = np.array(img)
    img[:,:,0] = img[:,:,0]-103.939
    img[:,:,1]=img[:,:,0]-116.779
    img[:,:,2]=img[:,:,0]-123.68
    immatrix.append(img)

from collections import Counter
print(Counter(imlabel))

# 훈련과 테스트 데이터 셋 생성
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(immatrix, imlabel, test_size = 0.2, random_state = 42, stratify=imlabel)

print(np.array(x_train).shape)
print(np.array(y_train).shape)
print(np.array(x_test).shape)
print(np.array(y_test).shape)
print(Counter(y_train))
print(Counter(y_test))

# One-Hot Encoding
from keras.utils import np_utils

y_train = np_utils.to_categorical(np.array(y_train),5)
y_test = np_utils.to_categorical(np.array(y_test), 5)

print(np.array(x_train).shape)
print(np.array(x_test).shape)
print(np.array(y_train).shape)
print(np.array(y_test).shape)

# CNN 모델 생성
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
    opt = Adam(lr=LR, decay=LR/ EPOCHS)    
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

lr = 0.0001
epochs = 10
classnum = 5
print(x_train[0].shape)
cnnmodel = createModel(lr, epochs, classnum,x_train[0].shape)
cnnmodel.summary()

#클래스 가중치 설정
class_weights={ 0: 1/((Counter(imlabel)[0]/len(imlabel))*100),
              1: 1/((Counter(imlabel)[1]/len(imlabel))*100),
              2: 1/((Counter(imlabel)[2]/len(imlabel))*100),
              3: 1/((Counter(imlabel)[3]/len(imlabel))*100),
              4: 1/((Counter(imlabel)[4]/len(imlabel))*100)}
print(class_weights)

#가중치 부여해서 실행
cnnmodel.fit(x_train, y_train, batch_size = 64, epochs=10, shuffle=True, verbose=2, class_weight=class_weights, validation_data=(x_test, y_test))

score = cnnmodel.evaluate(x_test, y_test, verbose=0)
print(score)
#
# #가중치 부여하지 않고 실행
# cnnmodel.fit(x_train, y_train, batch_size = 64, epochs=10, shuffle=True, verbose=2, validation_data=(x_test, y_test))
#
# # ResNet 모델 기반 전이학습
# import keras
# from keras.models import Model
# from keras.applications.resnet50 import ResNet50
# def createRresnetModel(LR, EPOCHS, NUM_CLASSES, inputShape, dim=224):
#   restnet = ResNet50(include_top=False, weights='imagenet', input_shape=inputShape)
#   output = restnet.layers[-1].output
#   output = keras.layers.Flatten()(output)
#   restnet = Model(restnet.input, output=output)
#   for layer in restnet.layers:
#       layer.trainable = False
#   restnet.summary()
#   model = Sequential()
#   model.add(restnet)
#   model.add(Dense(512, activation='relu', input_dim=inputShape))
#   model.add(Dropout(0.5))
#   model.add(Dense(512, activation='relu'))
#   model.add(Dropout(0.5))
#   model.add(Dense(activation='softmax', units=NUM_CLASSES))
#   opt = Adam(lr=LR, decay=LR/ EPOCHS)
#   model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#   return model
#
# lr = 0.0001
# epochs = 10
# classnum = 5
# resnetmodel = createRresnetModel(lr, epochs, classnum, x_train[0].shape)
# resnetmodel.summary()
#
# # add some visualization
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# SVG(model_to_dot(resnetmodel).create(prog='dot', format='svg'))
#
# #가중치 사용한 경우
# resnetmodel.fit(x_train, y_train, batch_size = 64, epochs=10, shuffle=True, verbose=2, class_weight=class_weights,validation_data=(x_test, y_test))
#
# #가중치 사용 안한 경우
# resnetmodel.fit(x_train, y_train, batch_size = 64, epochs=10, shuffle=True, verbose=2, validation_data=(x_test, y_test))
#
# score = resnetmodel.evaluate(x_train, y_train, verbose=0)
# print(score)
#
# # VGG16 모델을 이용한 전이학습
# import keras
# from keras.models import Model
# from keras.applications import VGG16
# def createVGG16Model(LR, EPOCHS, NUM_CLASSES, inputShape, dim=224):
#   vgg = VGG16(include_top=False, weights='imagenet', input_shape=inputShape)
#   output = vgg.layers[-1].output
#   output = keras.layers.Flatten()(output)
#   vgg = Model(vgg.input, output=output)
#   for layer in vgg.layers:
#       layer.trainable = False
#   vgg.summary()
#   model = Sequential()
#   model.add(vgg)
#   model.add(Dense(512, activation='relu', input_dim=inputShape))
#   model.add(Dropout(0.5))
#   model.add(Dense(512, activation='relu'))
#   model.add(Dropout(0.5))
#   model.add(Dense(activation='softmax', units=NUM_CLASSES))
#   opt = Adam(lr=LR, decay=LR/ EPOCHS)
#   model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#   return model
#
# lr = 0.0001
# epochs = 10
# classnum = 5
# vggmodel = createVGG16Model(lr, epochs, classnum, x_train[0].shape)
# vggmodel.summary()
#
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# SVG(model_to_dot(vggmodel).create(prog='dot', format='svg'))
#
# #가중치 사용한 경우
# vggmodel.fit(x_train, y_train, batch_size = 64, epochs=10, shuffle=True, verbose=2,class_weight=class_weights,validation_data=(x_test, y_test))
#
# #가중치 사용한 경우
# vggmodel.fit(x_train, y_train, batch_size = 64, epochs=10, shuffle=True, verbose=2,validation_data=(x_test, y_test))
#
# # Data augumentation
# from keras.preprocessing.image import ImageDataGenerator
# aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, \
#     height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,\
#     horizontal_flip=True, fill_mode="nearest")
# aug.fit(x_train)
#
# data_flow=aug.flow(x_train, y_train, batch_size=64)
#
# history=cnnmodel.fit_generator(data_flow, steps_per_epoch=100, epochs=10, validation_data=(x_test, y_test))
#
# result = cnnmodel.evaluate(x_test, y_test, batch_size=64)
# print(result)