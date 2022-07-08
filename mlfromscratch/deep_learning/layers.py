from __future__ import print_function, division
import math, numpy as np, copy
from mlfromscratch.deep_learning.activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU, TanH, ELU, SELU, Softmax

activation_functions = {'relu': ReLU, 'sigmoid': Sigmoid, 'selu': SELU, 'elu': ELU, 'softmax': Softmax, 'leaky_relu': LeakyReLU, 'tanh': TanH, 'softplus': SoftPlus}

class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()

class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = self.W = self.w0 = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W  = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        self.W_opt  = self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape + self.w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad):
        W = self.W
        if self.trainable:
            self.W = self.W_opt.update(self.W, self.layer_input.T.dot(accum_grad))
            self.w0 = self.w0_opt.update(self.w0, np.sum(accum_grad, axis=0, keepdims=True))
        return accum_grad.dot(W.T)

    def output_shape(self):
        return (self.n_units)

class RNN(Layer):
    def __init__(self, n_units, activation='tanh', bptt_trunc=5, input_shape=None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = activation_functions[activation]()
        self.trainable = True
        self.bptt_trunc = bptt_trunc
        self.W = self.V = self.U = None

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[1])
        self.U  = np.random.uniform(-limit, limit, (self.n_units, self.input_shape[1]))
        limit = 1 / math.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (self.input_shape[1], self.n_units))
        self.W = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.U_opt = self.V_opt = self.W_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape + self.U.shape + self.V.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        batch_size, timesteps, input_dim = X.shape
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps + 1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))
        self.states[:, -1] = np.zeros((batch_size, self.n_units))
        for t in range(timesteps):
            self.state_input[:, t] = X[:, t].dot(self.U.T) + self.states[:, t - 1].dot(self.W.T)
            self.states[:, t] = self.activation(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)
        return self.outputs

    def backward_pass(self, accum_grad):
        grad_U, grad_V, grad_W, accum_grad_next = np.zeros_like(self.U), np.zeros_like(self.V), np.zeros_like(self.W), np.zeros_like(accum_grad)
        for t in reversed(range(accum_grad.shape[1])):
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            grad_wrt_state = accum_grad[:, t].dot(self.V) * self.activation.gradient(self.state_input[:, t])
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_-1])
                grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.gradient(self.state_input[:, t_-1])
        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)
        return accum_grad_next

    def output_shape(self):
        return self.input_shape

class Conv2D(Layer):
    def __init__(self, n_filters, filter_shape, input_shape=None, padding='same', stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.trainable = True

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(np.prod(self.filter_shape))
        self.W = np.random.uniform(-limit, limit, size=(self.n_filters, self.input_shape[0], self.filter_shape[0], self.filter_shape[1]))
        self.w0 = np.zeros((self.n_filters, 1))
        self.W_opt = self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape + self.w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        self.X_col = image_to_column(X, self.filter_shape, self.stride, self.padding)
        self.W_col = self.W.reshape((self.n_filters, -1))
        return (self.W_col.dot(self.X_col) + self.w0).reshape(self.output_shape() + (X.shape[0])).transpose(3, 0, 1, 2)

    def backward_pass(self, accum_grad):
        accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)
        if self.trainable:
            self.W = self.W_opt.update(self.W, accum_grad.dot(self.X_col.T).reshape(self.W.shape))
            self.w0 = self.w0_opt.update(self.w0, np.sum(accum_grad, axis=1, keepdims=True))
        return column_to_image(self.W_col.T.dot(accum_grad), self.layer_input.shape, self.filter_shape, self.stride, self.padding)

    def output_shape(self):
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        return self.n_filters, int((self.input_shape[1] + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1), int((self.input_shape[2] + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1)

class BatchNormalization(Layer):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.trainable = True
        self.eps = 0.01
        self.running_mean = self.running_var = None

    def initialize(self, optimizer):
        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)
        self.gamma_opt = self.beta_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.gamma.shape + self.beta.shape)

    def forward_pass(self, X, training=True):
        if self.running_mean is None:
            self.running_mean = np.mean(X, axis=0)
            self.running_var = np.var(X, axis=0)
        if training and self.trainable:
            mean, var = np.mean(X, axis=0), np.var(X, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else: mean, var = self.running_mean, self.running_var
        self.X_centered = X - mean
        self.stddev_inv = 1 / np.sqrt(var + self.eps)
        return self.gamma * self.X_centered * self.stddev_inv + self.beta

    def backward_pass(self, accum_grad):
        gamma = self.gamma
        if self.trainable:
            self.gamma = self.gamma_opt.update(self.gamma, np.sum(accum_grad * self.X_centered * self.stddev_inv, axis=0))
            self.beta = self.beta_opt.update(self.beta, np.sum(accum_grad, axis=0))
        return gamma * self.stddev_inv * (accum_grad.shape[0] * accum_grad - np.sum(accum_grad, axis=0) - self.X_centered * self.stddev_inv**2 * np.sum(accum_grad * self.X_centered, axis=0)) / accum_grad.shape[0]

    def output_shape(self):
        return self.input_shape

class PoolingLayer(Layer):
    def __init__(self, pool_shape=(2, 2), stride=1, padding=0):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding
        self.trainable = True

    def forward_pass(self, X, training=True):
        self.layer_input = X
        batch_size, channels, height, width = X.shape
        _, out_height, out_width = self.output_shape()
        return self._pool_forward(image_to_column(X.reshape(batch_size * channels, 1, height, width), self.pool_shape, self.stride, self.padding)).reshape(out_height, out_width, batch_size, channels).transpose(2, 3, 0, 1)

    def backward_pass(self, accum_grad):
        channels, height, width = self.input_shape
        return column_to_image(self._pool_backward(accum_grad.transpose(2, 3, 0, 1).ravel()), (accum_grad.shape[0] * channels, 1, height, width), self.pool_shape, self.stride, 0).reshape((accum_grad.shape[0]) + self.input_shape)

    def output_shape(self):
        out_height = (self.input_shape[1] - self.pool_shape[0]) / self.stride + 1
        out_width = (self.input_shape[2] - self.pool_shape[1]) / self.stride + 1
        assert isinstance(out_height, int) and isinstance(out_width, int)
        return self.input_shape[0], int(out_height), int(out_width)

class MaxPooling2D(PoolingLayer):
    def _pool_forward(self, X_col):
        arg_max = np.argmax(X_col, axis=0).flatten()
        output = X_col[arg_max, range(arg_max.size)]
        self.cache = arg_max
        return output

    def _pool_backward(self, accum_grad):
        accum_grad_col = np.zeros((np.prod(self.pool_shape), accum_grad.size))
        accum_grad_col[self.cache, range(accum_grad.size)] = accum_grad
        return accum_grad_col

class AveragePooling2D(PoolingLayer):
    def _pool_forward(self, X_col):
        return np.mean(X_col, axis=0)

    def _pool_backward(self, accum_grad):
        accum_grad_col = np.zeros((np.prod(self.pool_shape), accum_grad.size))
        accum_grad_col[:, range(accum_grad.size)] = accum_grad / accum_grad_col.shape[0]
        return accum_grad_col

class ConstantPadding2D(Layer):
    def __init__(self, padding, padding_value=0):
        self.padding = padding
        self.trainable = True
        if not isinstance(padding[0], tuple): self.padding = ((padding[0], padding[0]), padding[1])
        if not isinstance(padding[1], tuple): self.padding = (self.padding[0], (padding[1], padding[1]))
        self.padding_value = padding_value

    def forward_pass(self, X, training=True):
        return np.pad(X, pad_width=((0,0), (0,0), self.padding[0], self.padding[1]), mode="constant", constant_values=self.padding_value)

    def backward_pass(self, accum_grad):
        (pad_top, _), (pad_left, _) = self.padding
        return accum_grad[:, :, pad_top:pad_top + self.input_shape[1], pad_left:pad_left + self.input_shape[2]]

    def output_shape(self):
        return self.input_shape[0], self.input_shape[1] + np.sum(self.padding[0]), self.input_shape[2] + np.sum(self.padding[1])

class ZeroPadding2D(ConstantPadding2D):
    def __init__(self, padding):
        self.padding = padding
        if isinstance(padding[0], int): self.padding = ((padding[0], padding[0]), padding[1])
        if isinstance(padding[1], int): self.padding = (self.padding[0], (padding[1], padding[1]))
        self.padding_value = 0

class Flatten(Layer):
    def __init__(self, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.prev_shape)

    def output_shape(self):
        return (np.prod(self.input_shape))

class UpSampling2D(Layer):
    def __init__(self, size=(2, 2), input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.size = size
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.repeat(self.size[0], axis=2).repeat(self.size[1], axis=3)

    def backward_pass(self, accum_grad):
        return accum_grad[:, :, ::self.size[0], ::self.size[1]]

    def output_shape(self):
        return self.input_shape[0], self.size[0] * self.input_shape[1], self.size[1] * self.input_shape[2]

class Reshape(Layer):
    def __init__(self, shape, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.shape = shape
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0]) + self.shape)

    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.prev_shape)

    def output_shape(self):
        return self.shape

class Dropout(Layer):
    def __init__(self, p=0.2):
        self.p = p
        self._mask = self.input_shape = self.n_units = None
        self.pass_through = self.trainable = True

    def forward_pass(self, X, training=True):
        c = 1 - self.p
        if training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            c = self._mask
        return X * c

    def backward_pass(self, accum_grad):
        return accum_grad * self._mask

    def output_shape(self):
        return self.input_shape

class Activation(Layer):
    def __init__(self, name):
        self.activation_name = name
        self.activation_func = activation_functions[name]()
        self.trainable = True

    def layer_name(self):
        return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape

def determine_padding(filter_shape, output_shape="same"):
    return (0, 0), (0, 0) if output_shape == "valid" or output_shape == "same" else (int(math.floor((filter_shape[0] - 1) / 2)), int(math.ceil((filter_shape[0] - 1) / 2))), (int(math.floor((filter_shape[1] - 1) / 2)), int(math.ceil((filter_shape[1] - 1) / 2)))
def get_im2col_indices(images_shape, filter_shape, padding, stride=1):
    out_height = int((images_shape[2] + np.sum(padding[0]) - filter_shape[0]) / stride + 1)
    out_width = int((images_shape[3] + np.sum(padding[1]) - filter_shape[1]) / stride + 1)
    i = np.tile(np.repeat(np.arange(filter_shape[0]), filter_shape[1]), images_shape[1]) + stride * np.repeat(np.arange(out_height), out_width)
    j = np.tile(np.arange(filter_shape[1]), filter_shape[0] * images_shape[1]) + stride * np.tile(np.arange(out_width), out_height)
    return np.repeat(np.arange(images_shape[1]), filter_shape[0] * filter_shape[1]).reshape(-1, 1), i.reshape(-1, 1), j.reshape(1, -1)
def image_to_column(images, filter_shape, stride, output_shape='same'):
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    return np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')[:, get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)].transpose(1, 2, 0).reshape(np.prod(filter_shape) * images.shape[1], -1)
def column_to_image(cols, images_shape, filter_shape, stride, output_shape='same'):
    (batch_size, channels, height, width), (pad_h, pad_w) = images_shape, determine_padding(filter_shape, output_shape)
    images_padded = np.zeros((batch_size, channels, height + np.sum(pad_h), width + np.sum(pad_w)))
    np.add.at(images_padded, (slice(None), get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)), cols.reshape(channels * np.prod(filter_shape), -1, batch_size).transpose(2, 0, 1))
    return images_padded[:, :, pad_h[0]:pad_h[0] + height, pad_w[0]:pad_w[0] + width]
