# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    d = len(train_set[0])
    W= np.zeros(d)
    b = 0
    iter = max_iter
    while iter != 0:
        for i in range(len(train_set)):
            yhat = np.sign(np.dot(train_set[i],W)+b)
            if yhat <= 0:
                yhat = 0
            else:
                yhat = 1
            W = W +learning_rate*(train_labels[i]-yhat)*train_set[i]
            b = b + learning_rate*(train_labels[i]-yhat)
        iter -= 1
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    
    W, b = trainPerceptron(train_set, train_labels,learning_rate, max_iter)
    result = []
    for i in range(len(dev_set)):
        yhat = np.sign(np.dot(dev_set[i],W)+b)
        if yhat <= 0:
            result.append(0)
        else:
            result.append(1)
    return result

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    ret_list = []
    for i in range(len(dev_set)):
        distances = []
        for j in range(len(train_set)):
            distance = np.linalg.norm(dev_set[i]-train_set[j])
            distances.append((distance,train_labels[j]))
        sorted_dist = sorted(distances)
        neighbors = sorted_dist[:k]
        t = 0
        f = 0
        for n in neighbors:
            label = n[1]
            if label == 1:
                t += 1
            else:
                f += 1
        if t > f:
            ret_list.append(1)
        else:
            ret_list.append(0)
    return ret_list
