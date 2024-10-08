import os
import sys
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, InputLayer, Conv2D, MaxPooling2D
from keras import applications
from keras import callbacks
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import math
import matplotlib.pyplot as plt
import cv2
from sklearn import preprocessing

image_size = (96,96) 
batch_size = 30
data_src = 'Datasets/' #pacbed data

data_images = []
data_labels = []
train_img = []
train_labels = []
val_img = []
val_labels = []
test_img = []
test_labels = []

#Data
folder = "SrPb3S/"
def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

for i in range(0,20): # noisy images
    counter = 0
    for t in range(0,200): # thickness (0-100 nm)
      #print(c_folders[c] + str(i) + str(t) + '\n')
      l = str(math.floor(counter)) # label = (thickness)
      path = data_src + folder + str(i) +'_'+ str(t) + '.npy'
      img = np.load(path)
      img = cv2.resize(img, dsize=image_size, interpolation=cv2.INTER_CUBIC) #resize so all images same size
      img = scale_range(img,0,1)
      img = img.astype(dtype=np.float32)
      img_size = img.shape[0]
      sx, sy = img.shape[0], img.shape[1]
      new_channel = np.zeros((img_size, img_size))
      img_stack = np.dstack((img, new_channel, new_channel))

      if i == 3:
        val_img.append(img_stack)
        val_labels.append(l)
      elif i == 4:
        test_img.append(img_stack)
        test_labels.append(l)
      else:
        train_img.append(img_stack)
        train_labels.append(l)

      counter += 0.5 #two dp images per thickness label, so update every 2 innermost loops



le = preprocessing.LabelEncoder()

le.fit(train_labels)
train_labels = le.transform(train_labels)
val_labels = le.transform(val_labels)

nb_train_samples = len(train_img)
nb_class = len(set(train_labels))
x_train = np.concatenate([arr[np.newaxis] for arr in train_img])
y_train = to_categorical(train_labels, num_classes=nb_class)
x_val = np.concatenate([arr[np.newaxis] for arr in val_img])
y_val = to_categorical(val_labels, num_classes=nb_class)
print('Size of image array in bytes')
print(x_train.nbytes)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',input_shape=x_train[0].shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(nb_class, activation='softmax'))



model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(),#lr=1e-3),
              metrics=['accuracy'])

model.summary()



model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(),#lr=1e-3),
              metrics=['accuracy'])

model.summary()


history = model.fit(x=x_train, y=y_train,
                    validation_data=(x_val, y_val),
                    batch_size=30,
                    epochs=30,
                    verbose=1)

model.save('STO-thickness.h5')

  


