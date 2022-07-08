from __future__ import division
import numpy as np
from mlfromscratch.utils import accuracy_score

class Loss(object):
    def loss(self, y_true, y_pred):
        raise NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return np.power(y - y_pred, 2) / 2

    def gradient(self, y, y_pred):
        return y - y_pred

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.log(p**y * (1 - p)**(1 - y))

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return (1 - y) / (1 - p)- y / p
