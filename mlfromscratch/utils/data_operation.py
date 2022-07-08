from __future__ import division
import numpy as np, math

def calculate_entropy(y):
    entropy = 0
    for label in np.unique(y):
        p = len(y[y == label]) / len(y)
        entropy -= p * math.log2(p)
    return entropy
def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))
def calculate_variance(X):
    mean = np.ones(np.shape(X)) * X.mean(0)
    return np.diag((X - mean).T.dot(X - mean)) / np.shape(X)[0]
def calculate_std_dev(X):
    return np.sqrt(calculate_variance(X))
def euclidean_distance(x1, x2):
    return math.dist(x1, x2)
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred, axis=0) / len(y_true)
def calculate_covariance_matrix(X, Y=None):
    if Y is None: Y = X
    return np.array((X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0)) / (np.shape(X)[0] - 1), dtype=float)
def calculate_correlation_matrix(X, Y=None):
    if Y is None: Y = X
    return np.array(np.divide((X - X.mean(0)).T.dot(Y - Y.mean(0)) / np.shape(X)[0], np.expand_dims(calculate_std_dev(X), 1).dot(np.expand_dims(calculate_std_dev(Y), 1).T)), dtype=float)
