# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 04:05:12 2019

@author: miru
"""
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
import pickle
from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_mnist

# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

print(min_pixel_value, max_pixel_value)
# pickle everything..
print("pickling pickling")
pickle_out = open("x_train.pickle", 'wb')
pickle.dump(x_train, pickle_out)
pickle_out.close()
pickle_out = open("y_train_ohe.pickle", 'wb')
pickle.dump(y_train, pickle_out)
pickle_out.close()
pickle_out = open("x_test.pickle", 'wb')
pickle.dump(x_test, pickle_out)
pickle_out.close()
pickle_out = open("y_test_ohe.pickle", 'wb')
pickle.dump(y_test, pickle_out)
pickle_out.close()
print("pickling done, abort")


im_shape = x_train[0].shape

# Step 2: Create the model

model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(23, 23, 4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])

# Step 3: Create the ART classifier

classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

# Step 4: Train the ART classifier

classifier.fit(x_train, y_train, batch_size=32, nb_epochs=10)
classifier.save('cnn_simple_mnist.model')
