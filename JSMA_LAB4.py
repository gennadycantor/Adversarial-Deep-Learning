# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 01:24:11 2019

@author: miru
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np
import matplotlib.pyplot as plt
from art.attacks import SaliencyMapMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset
import pickle


cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]

# load pre-trained model
model = tf.keras.models.load_model('cnn_cifar10.model')
model = KerasClassifier(model = model, clip_values = (0.0, 1.0))

# unpickle test data
pickle_in = open("x_test.pickle", 'rb')
test_data = pickle.load(pickle_in)
pickle_in.close()
#print(len(test_data))
print(test_data[0].shape, "shape shape yoyo")

pickle_in = open("y_test_ohe.pickle", 'rb')
test_labels = pickle.load(pickle_in)
pickle_in.close()
#print(len(test_labels))


"""
grab the first 100 guys for convenience
"""
def grabVictims():
    victimsList = []
    for i in range(100):
        victimsList.append(test_data[i])
    victims = np.array(victimsList)
    return victims

"""
generate adversarial examples with jacobian saliency
"""
def JSMA(victims):
    adv_crafter = SaliencyMapMethod(model)
    finalVictims = adv_crafter.generate(x = victims)
    return finalVictims

"""
test accuracy of non-adversarial examples
and then plot the first 9 guys
"""
def testAccuracyBenign(clean_victims):
    predictions = model.predict(clean_victims)
    x_test_pred = np.argmax(model.predict(clean_victims[:100]), axis=1)
    # sum up every correct instances 
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(test_labels[0:100], axis=1)) / len(test_labels[0:100])
    print('Accuracy on benign test examples: {}%'.format(accuracy * 100))
    # set figsize
    plt.figure(figsize=(10,10))
    # just print the first 9
    for i in range(0, 9):
        # get predicted integer class and true integer class
        pred_label, true_label = cifar10_classes[x_test_pred[i]], cifar10_classes[np.argmax(test_labels[i])]
        # add subplots: 3 = num rows, 3 = columns, 1 + i = sweep thorugh 1 -> 9
        plt.subplot(330 + 1 + i)
        # if for any rzn, you are doing black and white (like mnist) also pass: cmap = 'binary' in
        # as argument
        fig=plt.imshow(clean_victims[i])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # true labels to the right - in () - of predicted labels
        fig.axes.text(0.5, -0.1, pred_label + " (" + true_label + ")", fontsize=12, transform=fig.axes.transAxes, 
                      horizontalalignment='center')

"""
test accuracy of adversarial examples
and then plot the first 9 guys
"""
def testAccuracyContaminated(dirty_victims):
    x_test_adv_pred = np.argmax(model.predict(dirty_victims), axis=1)
    predictions = model.predict(dirty_victims)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(test_labels[0:100], axis=1)) / len(test_labels[0:100])
    print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
    plt.figure(figsize=(10,10))
    for i in range(0, 9):
        pred_label, true_label = cifar10_classes[x_test_adv_pred[i]], cifar10_classes[np.argmax(test_labels[i])]
        plt.subplot(330 + 1 + i)
        fig=plt.imshow(dirty_victims[i])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        fig.axes.text(0.5, -0.1, pred_label + " (" + true_label + ")", fontsize=12, transform=fig.axes.transAxes, 
                      horizontalalignment='center')

def main():
    # get "clean victims"
    victims = grabVictims()
    #print(victims.shape)
    #print(test_data.shape)
    # get adversarial examples
    adversarialExamples = JSMA(victims)
    #print(len(adversarialExamples))
    testAccuracyBenign(victims)
    testAccuracyContaminated(adversarialExamples)
    

main()

