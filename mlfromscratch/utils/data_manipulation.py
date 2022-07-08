from __future__ import division
from itertools import combinations_with_replacement
import numpy as np

def shuffle_data(X, y, seed=None):
    if seed: np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]
def batch_iterator(X, y=None, batch_size=64):
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        yield X[begin:end], y[begin:end] if y is not None else X[begin:end]
def divide_on_feature(X, feature_i, threshold):
    split_func = lambda sample: sample[feature_i] >= threshold if isinstance(threshold, int) or isinstance(threshold, float) else lambda sample: sample[feature_i] == threshold
    return np.array([np.array([sample for sample in X if split_func(sample)]), np.array([sample for sample in X if not split_func(sample)])])
def polynomial_features(X, degree):
    combinations = [item for sublist in [combinations_with_replacement(range(np.shape(X)[1]), i) for i in range(degree + 1)] for item in sublist]
    X_new = np.empty((np.shape(X)[0], len(combinations)))
    for i, index_combs in enumerate(combinations): X_new[:, i] = np.prod(X[:, index_combs], axis=1)
    return X_new
def get_random_subsets(X, y, n_subsets, replacements=True):
    n_samples = np.shape(X)[0]
    X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets, subsample_size = [], int(n_samples // 2)
    if replacements: subsample_size = n_samples
    for _ in range(n_subsets):
        idx = np.random.choice(range(n_samples), size=np.shape(range(subsample_size)), replace=replacements)
        subsets.append([X_y[idx][:, :-1], X_y[idx][:, -1]])
    return subsets
def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)
def standardize(X):
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]: X[:, col] = (X[:, col] - X.mean(axis=0)[col]) / std[col]
    return X
def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    if shuffle: X, y = shuffle_data(X, y, seed)
    split_i = len(y) - int(len(y) // (1 / test_size))
    return X[:split_i], X[split_i:], y[:split_i], y[split_i:]
def k_fold_cross_validation_sets(X, y, k, shuffle=True):
    if shuffle: X, y = shuffle_data(X, y)
    left_overs, n_left_overs = {}, len(y) % k
    if n_left_overs != 0:
        left_overs["X"] = X[-n_left_overs:]
        left_overs["y"] = y[-n_left_overs:]
        X = X[:-n_left_overs]
        y = y[:-n_left_overs]
    X_split, y_split, sets = np.split(X, k), np.split(y, k), []
    for i in range(k): sets.append([np.concatenate(X_split[:i] + X_split[i + 1:], axis=0), X_split[i], np.concatenate(y_split[:i] + y_split[i + 1:], axis=0), y_split[i]])
    if n_left_overs != 0:
        np.append(sets[-1][0], left_overs["X"], axis=0)
        np.append(sets[-1][2], left_overs["y"], axis=0)
    return np.array(sets)
def to_categorical(x, n_col=None):
    if not n_col: n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot
def to_nominal(x):
    return np.argmax(x, axis=1)
def make_diagonal(x):
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])): m[i, i] = x[i]
    return m
