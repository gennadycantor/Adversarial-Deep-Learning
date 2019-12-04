from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np
import pickle
from art.attacks import DeepFool
from art.classifiers import KerasClassifier
from art.utils import load_dataset


# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('cifar10'))
print(min_, "min")
print(max_, "max")
print(len(x_train))
print(len(x_test))
x_train, y_train = x_train[:50000], y_train[:50000]
x_test, y_test = x_test[:10000], y_test[:10000]

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

# Create Keras convolutional neural network
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier = KerasClassifier(model=model, clip_values=(min_, max_))
classifier.fit(x_train, y_train, nb_epochs=10, batch_size=64)
#classifier.save('cnn_mnist.model')