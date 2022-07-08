from __future__ import print_function, division
import numpy as np, math
from sklearn import datasets
from mlfromscratch.utils import train_test_split, to_categorical, normalize, accuracy_score, Plot
from mlfromscratch.deep_learning.activation_functions import Sigmoid, Softmax
from mlfromscratch.deep_learning.loss_functions import CrossEntropy

class MultilayerPerceptron():
    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.01):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()

    def _initialize_weights(self, X, y):
        limit = 1 / math.sqrt(X.shape[1])
        self.W  = np.random.uniform(-limit, limit, (X.shape[1], self.n_hidden))
        self.w0 = np.zeros((1, self.n_hidden))
        limit = 1 / math.sqrt(self.n_hidden)
        self.V  = np.random.uniform(-limit, limit, (self.n_hidden, y.shape[1]))
        self.v0 = np.zeros((1, y.shape[1]))

    def fit(self, X, y):
        self._initialize_weights(X, y)
        for _ in range(self.n_iterations):
            hidden_input = X.dot(self.W) + self.w0
            hidden_output = self.hidden_activation(hidden_input)
            output_layer_input = hidden_output.dot(self.V) + self.v0
            grad_wrt_out_l_input = self.loss.gradient(y, self.output_activation(output_layer_input)) * self.output_activation.gradient(output_layer_input)
            grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.V.T) * self.hidden_activation.gradient(hidden_input)
            self.V -= self.learning_rate * hidden_output.T.dot(grad_wrt_out_l_input)
            self.v0 -= self.learning_rate * np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)
            self.W -= self.learning_rate * X.T.dot(grad_wrt_hidden_l_input)
            self.w0 -= self.learning_rate * np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

    def predict(self, X):
        return self.output_activation(self.hidden_activation(X.dot(self.W) + self.w0).dot(self.V) + self.v0)

data = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(normalize(data.data), to_categorical(data.target), test_size=0.4, seed=1)
clf = MultilayerPerceptron(16, 1000)
clf.fit(X_train, y_train)
y_pred = np.argmax(clf.predict(X_test), axis=1)
accuracy = accuracy_score(np.argmax(y_test, 1), y_pred)
print ("Accuracy:", accuracy)
Plot().plot_in_2d(X_test, y_pred, "Multilayer Perceptron", accuracy, np.unique(to_categorical(data.target)))
