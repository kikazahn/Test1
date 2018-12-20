# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 13:26:27 2018

@author: Kirill
"""
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from tensorflow import set_random_seed

set_random_seed(100)
np.random.seed(10)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

X_train1 = X_train[0:40000]
X_train2 = X_train[20000:60000]
X_train3 = np.concatenate((X_train[0:20000],X_train[40000:60000]))
Y_train1 = Y_train[0:40000]
Y_train2 = Y_train[20000:60000]
Y_train3 = np.concatenate((Y_train[0:20000],Y_train[40000:60000]))

XTRAIN = (X_train1, X_train2, X_train3)
YTRAIN = (Y_train1, Y_train2, Y_train3)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)

model = Sequential()
model.add(Convolution2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
#model.add(Convolution2D(256, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(Dense(128,activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

names = ('model1', 'model2', 'model3')
path = 'D:/MTS_WORK/MNIST/Ensemble/'

for i in range(3):
    
    model.fit(XTRAIN[i], YTRAIN[i], batch_size=100,epochs=12,verbose=1,callbacks=[learning_rate_reduction],
          validation_data = (X_test,Y_test))
    
    model.save(path + names[i])
    score = model.evaluate(X_test, Y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


