# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 04:26:14 2019

@author: miru
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np
import matplotlib.pyplot as plt
from art.attacks import SaliencyMapMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset
import pickle

mnist_classes = ['zero', 'one', 'two', 'three', 'four', 
                 'five', 'six', 'seven', 'eight', 'nine']

# load pre-trained model
model = tf.keras.models.load_model('cnn_simple_mnist.model')
model = KerasClassifier(model = model, clip_values = (0.0, 1.0))
#set targets
target_airplane = np.zeros(10, dtype=np.float)
target_airplane[0] = 1
target_airplane = np.reshape(target_airplane, (10,))
target_truck = np.zeros(10, dtype=np.float)
target_truck[9] = 1
target_truck = np.reshape(target_truck, (10,))

# unpickle test data
pickle_in = open("x_test.pickle", 'rb')
test_data = pickle.load(pickle_in)
pickle_in.close()
#print(len(test_data))

pickle_in = open("y_test_ohe.pickle", 'rb')
test_labels = pickle.load(pickle_in)
pickle_in.close()
#print(len(test_labels))


def grabVictims():
    victimsList = []
    for i in range(100):
        victimsList.append(test_data[i])
    victims = np.array(victimsList)
    return victims

def JSMA(victims):
    adv_crafter = SaliencyMapMethod(model)
    finalVictims = adv_crafter.generate(x = victims)
    return finalVictims

def testAccuracyBenign(clean_victims):
    predictions = model.predict(clean_victims)
    x_test_pred = np.argmax(model.predict(clean_victims[:100]), axis=1)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(test_labels[0:100], axis=1)) / len(test_labels[0:100])
    print('Accuracy on benign test examples: {}%'.format(accuracy * 100))
    plt.figure(figsize=(10,10))
    for i in range(0, 9):
        pred_label, true_label = mnist_classes[x_test_pred[i]], mnist_classes[np.argmax(test_labels[i])]
        plt.subplot(330 + 1 + i)
        fig=plt.imshow(cv2.resize(clean_victims[i], (28,28)), cmap = 'binary')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        fig.axes.text(0.5, -0.1, pred_label + " (" + true_label + ")", fontsize=12, transform=fig.axes.transAxes, 
                      horizontalalignment='center')
    
def testAccuracyContaminated(dirty_victims):
    x_test_adv_pred = np.argmax(model.predict(dirty_victims), axis=1)
    predictions = model.predict(dirty_victims)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(test_labels[0:100], axis=1)) / len(test_labels[0:100])
    print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
    plt.figure(figsize=(10,10))
    for i in range(0, 9):
        pred_label, true_label = mnist_classes[x_test_adv_pred[i]], mnist_classes[np.argmax(test_labels[i])]
        plt.subplot(330 + 1 + i)
        fig=plt.imshow(cv2.resize(dirty_victims[i], (28,28)), cmap = 'binary')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        fig.axes.text(0.5, -0.1, pred_label + " (" + true_label + ")", fontsize=12, transform=fig.axes.transAxes, 
                      horizontalalignment='center')
        
def testAccuracySnapped(snapped_victims):
    predictions = model.predict(snapped_victims)
    x_test_pred = np.argmax(model.predict(snapped_victims), axis=1)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(test_labels[0:100], axis=1)) / len(test_labels[0:100])
    print('Accuracy on snapped test examples: {}%'.format(accuracy * 100))
    plt.figure(figsize=(10,10))
    for i in range(0, 9):
        pred_label, true_label = mnist_classes[x_test_pred[i]], mnist_classes[np.argmax(test_labels[i])]
        plt.subplot(330 + 1 + i)
        fig=plt.imshow(cv2.resize(snapped_victims[i], (28,28)), cmap = 'binary')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        fig.axes.text(0.5, -0.1, pred_label + " (" + true_label + ")", fontsize=12, transform=fig.axes.transAxes, 
                      horizontalalignment='center')
        
def snap(test_data, snap_interval):
    divisor = 255/snap_interval
    test_data = np.divide(test_data, divisor)
    test_data = np.around(test_data)
    test_data = test_data * divisor
    return test_data

def prepare(snappedExamples):
    snappedExamplesList = []
    for i in range(len(snappedExamples)):
        new_array = np.reshape(snappedExamples[i], (-1,28,28,1))
        snappedExamplesList.append(new_array)
    return np.array(snappedExamples)

def main():
    victims = grabVictims()
    pickle_out = open("victims.pickle", 'wb')
    pickle.dump(victims, pickle_out)
    pickle_out.close()
    adversarialExamples = JSMA(victims)
    snappedExamples = snap(adversarialExamples, 215)
    snappedExamples = prepare(snappedExamples)
    pickle_out = open("adversararialExamplesJSMA.pickle", 'wb')
    pickle.dump(adversarialExamples, pickle_out)
    pickle_out.close()
    testAccuracyBenign(victims)
    testAccuracyContaminated(adversarialExamples)
    testAccuracySnapped(snappedExamples)
    

main()