from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt, numpy as np
from mlfromscratch.deep_learning import NeuralNetwork
from mlfromscratch.utils import train_test_split, to_categorical, Plot
from mlfromscratch.deep_learning.optimizers import Adam
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.deep_learning.layers import Dense, Dropout, Conv2D, Flatten, Activation, BatchNormalization

data = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, to_categorical(data.target.astype('int')), 0.4, seed=1)
X_test = X_test.reshape((-1, 1, 8, 8))
clf = NeuralNetwork(Adam(), CrossEntropy, (X_test, y_test))
clf.add(Conv2D(16, (3, 3), (1, 8, 8)))
clf.add(Activation('relu'))
clf.add(Dropout(0.25))
clf.add(BatchNormalization())
clf.add(Conv2D(32, (3, 3)))
clf.add(Activation('relu'))
clf.add(Dropout(0.25))
clf.add(BatchNormalization())
clf.add(Flatten())
clf.add(Dense(256))
clf.add(Activation('relu'))
clf.add(Dropout(0.4))
clf.add(BatchNormalization())
clf.add(Dense(10))
clf.add(Activation('softmax'))
clf.summary(name="ConvNet")
train_err, val_err = clf.fit(X_train.reshape((-1, 1, 8, 8)), y_train, n_epochs=50, batch_size=256)
plt.legend(handles=[plt.plot(range(len(train_err)), train_err, label="Training Error")[0], plt.plot(range(len(train_err)), val_err, label="Validation Error")[0]])
plt.title("Error Plot")
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.show()
accuracy = clf.test_on_batch(X_test, y_test)[1]
print ("Accuracy:", accuracy)
Plot().plot_in_2d(X_test.reshape(-1, 64), np.argmax(clf.predict(X_test), axis=1), "Convolutional Neural Network", accuracy, range(10))
