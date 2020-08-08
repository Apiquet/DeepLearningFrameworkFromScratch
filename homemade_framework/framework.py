#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script implements a Deep Learning framework
with the types of classes stored in __types__ variable.
It also implements features provided by a Deep Learning Framework such as:
saving and loading a model to deploy it, getting its number of parameters,
drawing learning curves, printing its description.
"""

__types__ = ["Linear", "Activation", "Loss",
             "Softmax", "Flatten", "Convolution",
             "BatchNormalization", "MaxPooling2D",
             "AveragePooling2D"]


from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np


def print_current_results(epochs, model, train_features, train_target,
                          test_features, test_target, loss_sum, prefix=""):
    """Compute and save train/test error, print with epoch number and loss.

    Keyword arguments:
    epochs -- current epoch number
    Model -- neural network model
    train_features -- features of the train set
    train_target -- target of the train set
    test_features -- features of the test set
    test_target -- target of the test set
    loss_sum -- current loss
    prefix -- optional argument to add a prefix
    """

    train_error = compute_accuracy(model, train_features, train_target)
    test_error = compute_accuracy(model, test_features, test_target)
    model.train_error.append(train_error)
    model.test_error.append(test_error)
    print(prefix + "Epoch: {}, Train Error: {:.4f}%,\
        Test Error: {:.4f}%, Loss  {:.4f}".format(epochs, train_error,
                                                  test_error, loss_sum))


def print_in_color(message, color="red"):
    """Print any message in color for Jupyter cell.

    Keyword arguments:
    message -- text to print
    color -- color wanted in choices dictionary
    """

    choices = {"green": "32", "blue": "34",
               "magenta": "35", "red": "31",
               "Gray": "37", "Cyan": "36",
               "Black": "39", "Yellow": "33",
               "highlight": "40"}
    if message == "-h":
        return list(choices.keys())
    elif color == "":
        print(message)
    elif color in choices:
        print("\033[" + choices[color] + "m" + message + "\033[0m")
    else:
        raise ValueError("Available colors: {}, '-h' to get\
            the list".format(choices.keys()))


def train_homemade_model(model, num_epochs, train_features,
                         train_target, test_features,
                         test_target, batch_size, print_every_n_epochs=1):
    """Train a model in mini-batch and print its results.

    Keyword arguments:
    model -- neural network model
    num_epochs -- number of epochs to do
    train_features -- features of the train set
    train_target -- target of the train set
    test_features -- features of the test set
    test_target -- target of the test set
    batch_size -- batch size used for mini-batch
    """

    start_time = datetime.now()
    # Convert train_target to one hot encoding
    train_target_one_hot = convert_to_one_hot_labels(train_target)

    print_current_results(0, model, train_features, train_target,
                          test_features, test_target, 0,
                          prefix="Before training: ")
    test_results = []
    for epochs in range(0, num_epochs):
        loss_sum = 0
        test_results.append(get_inferences(model, test_features))
        for b in range(train_features.shape[0] // batch_size):
            output = model.forward(train_features[
                list(range(b*batch_size, (b+1)*batch_size))])
            loss = model.backward(train_target_one_hot[
                list(range(b*batch_size, (b+1)*batch_size))],
                                  output)
            loss_sum = loss_sum + loss.item()
        if epochs % print_every_n_epochs == 0:
            print_current_results(epochs + 1, model, train_features,
                                  train_target, test_features,
                                  test_target, loss_sum)

    training_time = datetime.now() - start_time
    print('\nTraining time: {}'.format(training_time))
    print_current_results(epochs, model, train_features, train_target,
                          test_features, test_target, loss_sum,
                          prefix="After training: ")


def learning_curves(model):
    """Plot train and test accuracy curves.

    Keyword arguments:
    model -- neural network model
    """

    epochs = range(len(model.train_error))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(epochs, model.train_error, 'b', label='Training error')
    plt.plot(epochs, model.test_error, 'r', label='Validation error')
    plt.title('Training and Validation error')
    plt.xlabel('Epochs')
    plt.ylabel('Error percentage')
    ax.set_yticks(np.arange(0,100,5))
    plt.grid()
    plt.legend()
    plt.show()


# Data Manager
def generate_disc_set(nb):
    """Generate dataset with nb number of samples.
    Features are tuples (i,j) with i,j in [-1,1]
    Target is 1 if (i^2 + j^2)/pi < 0.2, 0 otherwise
    This dataset allows to train a network on a non-linear task, a circle

    Keyword arguments:
    nb -- number of samples to generate

    Return: list of features and targets
    """

    features = np.random.uniform(-1, 1, (nb, 2))
    target = (features[:, 0]**2 + features[:, 1]**2)/math.pi < 0.2
    return features, target.astype(int)


def plot_dataset(features, target):
    """Create the plot for the dataset generated by generate_disc_set.

    Keyword arguments:
    features -- n tuples in [-1, 1]
    features -- n number (0 or 1)

    Return: plot object
    """

    fig, ax = plt.subplots(figsize=(4, 4))
    plt.title("Dataset")
    scatter = ax.scatter(features[:, 0], features[:, 1],
                         c=target)
    legend = ax.legend(*scatter.legend_elements(), title="Labels",
                       loc="lower right")
    ax.add_artist(legend)
    return plt


def convert_to_one_hot_labels(target):
    """Convert targets to one-hot-labels.
    For instance, if targets can be 0 or 1,
    all 0 will be converted to 01 and 1 to 10

    Keyword arguments:
    target -- list of all the targets

    Return: targets converted to one-hot-label
    """

    n_values = max(target) + 1
    target_onehot = np.eye(n_values)[target]
    return target_onehot


def compute_accuracy(model, data_features, data_targets):
    """Calculate error results in pourcentage of a model

    Keyword arguments:
    model -- neural network model
    data_features -- features set
    data_targets -- targets set

    Return: error results in pourcentage
    """

    predicted_classes = get_inferences(model, data_features)
    nb_data_errors = sum(data_targets != predicted_classes)
    return nb_data_errors/data_features.shape[0]*100


def get_inferences(model, data_features):
    """Compute inference of a model

    Keyword arguments:
    model -- neural network model
    data_features -- features set

    Return: list of predictions
    """

    output = model.forward(data_features)
    predicted_classes = np.argmax(output, axis=1)
    return predicted_classes


class Module(object):
    """Heritage class definition for all the framework's objects

    Public methods:
    __init__ -- initiate class attributes
    forward -- apply the class forward pass
    backward -- apply the class backward pass
    update -- update the internal weights
    print -- print class description
    getParametersCount -- return number of parameters of the class
    save -- save the class weights
    load -- load weights to deploy a model
    """

    def __init__(self):
        super().__init__()

    def forward(self, *x):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def update(self, *gradwrtoutput):
        raise NotImplementedError

    def print(self):
        raise NotImplementedError

    def getParametersCount(self):
        return 0

    def save(self, path, i):
        return

    def load(self, path, i):
        return


class ReLU(Module):
    """Activation class: ReLU

    Public methods:
    __init__ -- initiate class attributes
    forward -- OUT = IN if in > 0, else 0
    backward -- OUT = 1*IN if IN was > 0, else 0
    print -- print class description
    """

    def __init__(self):
        super().__init__()
        self.type = "Activation"
        self.name = "ReLU"
        self.prev_x = 0

    def forward(self, x):
        self.prev_x = x
        x[x < 0] = 0
        y = x
        return y

    def backward(self, dout):
        y = (self.prev_x > 0).astype(float)
        return np.multiply(y, dout)

    def print(self, color=""):
        print_in_color("\tReLU activation", color)
        return


class LeakyReLU(Module):
    """Activation class: Leaky ReLU

    Public methods:
    __init__ -- set alpha attribute, default: 0.01
    forward -- OUT = IN if in >= 0, else OUT = alpha*IN
    backward -- OUT = 1*IN if IN was >= 0, else OUT = alpha*IN
    print -- print class description
    """

    def __init__(self, a=0.01):
        super().__init__()
        self.type = "Activation"
        self.name = "LeakyReLU"
        self.prev_x = 0
        self.a = a

    def forward(self, x):
        self.prev_x = x
        neg = x < 0
        pos = x >= 0
        y = np.multiply(neg.astype(float), x)*self.a +\
            np.multiply(pos.astype(float), x)
        return y

    def backward(self, dout):
        neg = self.prev_x < 0
        pos = self.prev_x >= 0
        y = np.multiply(neg.astype(float), dout)*self.a +\
            np.multiply(pos.astype(float), dout)
        return y

    def print(self, color=""):
        print_in_color("\tLeakyReLU activation, a={}".format(self.a), color)
        return


class Sigmoid(Module):
    """Activation class: Sigmoid

    Public methods:
    __init__ -- initiate class attributes
    eq -- apply Sigmoid on given data
    forward -- OUT = 1 / (1 + exp(-IN)), IN has to be saved for backward pass
    backward -- OUT = [Sigmoid(forward_IN) * (1 - Sigmoid(forward_IN))] * IN
    print -- print class description
    """

    def __init__(self):
        super().__init__()
        self.type = "Activation"
        self.name = "Sigmoid"
        self.prev_x = 0

    def eq(self, x):
        return 1 / (1 + np.exp(np.multiply(x, -1)))

    def forward(self, x):
        self.prev_x = x
        y = self.eq(x)
        return y

    def backward(self, x):
        y = np.multiply(self.eq(self.prev_x) * (1 - self.eq(self.prev_x)), x)
        return y

    def print(self, color=""):
        print_in_color("\tSigmoid activation", color)
        return


class LossMSE(Module):
    """Loss MSE implementation

    Public methods:
    __init__ -- initiate class attributes
    loss -- OUT = 1/N * SUM((y_predicted - y)^2), with N the number of samples
    derivative -- OUT = 2*(y_predicted-y)/N, with N the number of samples
    print -- print class description
    """

    def __init__(self):
        super().__init__()
        self.type = "Loss"
        self.name = "LossMSE"

    def loss(self, y, y_pred):
        loss = sum(((y_pred - y)**2).sum(axis=0))/y.shape[1]
        return loss

    def derivative(self, y, y_pred):
        return 2*(y_pred-y)/y.shape[1]

    def print(self, color=""):
        print_in_color("\tMSE", color)


class Softmax(Module):
    """Softmax implementation

    Public methods:
    __init__ -- initiate class attributes
    eq -- apply Softmax on given data
    forward -- OUT = exp(IN_i)/exp(sum(IN)), IN saved for backward pass
    backward -- OUT = [Softmax(forward_IN) * (1 - Softmax(forward_IN))] * IN
    print -- print class description
    """

    def __init__(self):
        super().__init__()
        self.type = "Softmax"
        self.prev_x = 0

    def eq(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1)[:, None]

    def forward(self, x):
        self.prev_x = x
        y = self.eq(x)
        return y

    def backward(self, x):
        y = np.multiply(self.eq(self.prev_x) * (1 - self.eq(self.prev_x)), x)
        return y

    def print(self, color=""):
        print_in_color("\tSoftmax function", color)
        return


class BatchNorm(Module):
    """Batch Normalization implementation from kratzert on github.io
    Thank you kratzert for his clear impl. of the forward and backward passes,
    I could not do one more readable.
    Note: this is the only class which comes from online resources

    Public methods:
    __init__ -- initiate class attributes
    forward -- achieve forward pass
    backward -- achieve backward pass and update gamma and beta
    update -- update gamma and beta with given parameters
    set_Lr -- set learning rate used to update gamma and beta
    getParametersCount -- return 2, for gamma and beta
    save -- save gamma and beta to BatchNormalization-N-gamma/beta.bin files
    load -- load gamma/beta values from previous training to deploy the model
    print -- print class description
    """

    def __init__(self):
        super().__init__()
        self.type = "BatchNormalization"
        self.gamma = 1
        self.eps = 10**-100
        self.beta = 0

    def forward(self, x):
        N, D = x.shape

        # step1: calculate mean
        mu = 1./N * np.sum(x, axis=0)

        # step2: subtract mean vector of every trainings example
        self.xmu = x - mu

        # step3: following the lower branch - calculation denominator
        sq = self.xmu ** 2

        # step4: calculate variance
        self.var = 1./N * np.sum(sq, axis=0)

        # step5: add eps for numerical stability, then sqrt
        self.sqrtvar = np.sqrt(self.var + self.eps)

        # step6: invert sqrtwar
        self.ivar = 1./self.sqrtvar

        # step7: execute normalization
        self.xhat = self.xmu * self.ivar

        # step8: Nor the two transformation steps
        gammax = self.gamma * self.xhat

        # step9
        out = gammax + self.beta
        return out

    def backward(self, dout):
        N, D = dout.shape

        # step9
        dbeta = np.sum(dout, axis=0)
        dgammax = dout  # not necessary, but more understandable

        # step8
        dgamma = np.sum(dgammax*self.xhat, axis=0)
        self.update(dgamma, dbeta)
        dxhat = dgammax * self.gamma

        # step7
        divar = np.sum(dxhat*self.xmu, axis=0)
        dxmu1 = dxhat * self.ivar

        # step6
        dsqrtvar = -1. / (self.sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / np.sqrt(self.var + self.eps) * dsqrtvar

        # step4
        dsq = 1. / N * np.ones((N, D)) * dvar

        # step3
        dxmu2 = 2 * self.xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

        # step1
        dx2 = 1. / N * np.ones((N, D)) * dmu

        # step0
        dx = dx1 + dx2
        return dx

    def update(self, dgamma, dbeta):
        self.gamma = self.gamma - self.lr * np.mean(dgamma)
        self.beta = self.beta - self.lr * np.mean(dbeta)

    def set_Lr(self, lr):
        self.lr = lr
        return

    def getParametersCount(self):
        return 2

    def save(self, path, i):
        with open(path + self.type + i + '-gamma.bin', "wb") as f:
            self.gamma.tofile(f)
        with open(path + self.type + i + '-beta.bin', "wb") as f:
            self.beta.tofile(f)
        return [self.type, self.gamma, self.beta]

    def load(self, path, i):
        with open(path + self.type + i + '-gamma.bin', "rb") as f:
            self.gamma = np.fromfile(f)
        with open(path + self.type + i + '-beta.bin', "rb") as f:
            self.beta = np.fromfile(f)

    def print(self, color=""):
        print_in_color("\tBatch normalization function: a={}, b={}".format(
            self.gamma, self.beta), color)
        return


class Linear(Module):
    """Linear layer implementation

    Public methods:
    __init__ -- init IN/OUT sizes, Xavier initialization for weights and bias
    forward -- OUT = IN * Weights + Bias
    backward -- update Weights and Bias, return IN * Weights
    update -- update weights and bias with given parameters
    set_Lr -- set learning rate used to update weights and bias
    getParametersCount -- return number of weights + number of bias
    save -- save weights and bias to Linear-N-weights/bias.bin files
    load -- load weights and bias from previous training to deploy the model
    print -- print class description
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.type = "Linear"
        self.prev_x = np.zeros(out_features)
        self.in_features = in_features
        self.out_features = out_features
        stdv = 1. / math.sqrt(self.out_features)
        self.weight = np.random.uniform(-stdv, stdv, (self.in_features,
                                                      self.out_features))
        self.bias = np.random.uniform(-stdv, stdv, (self.out_features, 1))

    def update(self, dout):
        lr = self.lr
        self.weight = self.weight -\
            np.multiply(lr, np.matmul(np.transpose(self.prev_x), dout))
        self.bias = self.bias -\
            lr*dout.mean(0).reshape([self.bias.shape[0], 1])*1

    def backward(self, dout):
        b = np.matmul(dout, np.transpose(self.weight))
        self.update(dout)
        return b

    def forward(self, x):
        self.prev_x = x
        return np.matmul(x, self.weight) +\
            np.transpose(np.repeat(self.bias, x.shape[0], axis=1))

    def set_Lr(self, lr):
        self.lr = lr
        return

    def getParametersCount(self):
        return np.prod(self.weight.shape) + np.prod(self.bias.shape)

    def save(self, path, i):
        with open(path + self.type + i + '-weights.bin', "wb") as f:
            self.weight.tofile(f)
        with open(path + self.type + i + '-bias.bin', "wb") as f:
            self.bias.tofile(f)

    def load(self, path, i):
        with open(path + self.type + i + '-weights.bin', "rb") as f:
            self.weight = np.fromfile(f).reshape([self.in_features,
                                                  self.out_features])
        with open(path + self.type + i + '-bias.bin', "rb") as f:
            self.bias = np.fromfile(f).reshape([self.out_features, 1])

    def print(self, color=""):
        msg = "\tLinear layer shape: {}".format([self.weight.shape[0],
                                                 self.weight.shape[1]])
        print_in_color(msg, color)

    def print_weight(self):
        print(self.weight)


class Convolution(Module):
    """Convolution layer implementation

    Public methods:
    __init__ -- init IN/OUT sizes, Xavier initialization for weights and bias
    convolution -- apply a simple convolution
    forward -- apply convolution and save needed data for the backward pass
    backward -- update Weights and Bias, return dout wrt input
    update -- update weights and bias with given parameters
    set_Lr -- set learning rate used to update weights and bias
    getParametersCount -- return number of weights + number of bias
    save -- save weights and bias to Convolution-N-weights/bias.bin files
    load -- load weights and bias from previous training to deploy the model
    print -- print class description
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=0, padding=0):
        super().__init__()
        self.type = "Convolution"
        self.k_height = kernel_size
        self.k_width = kernel_size
        self.x_width = 0
        self.x_height = 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        stdv = 1. / math.sqrt(self.k_height)
        self.kernel = np.random.uniform(-stdv, stdv, (self.out_channels,
                                                      self.in_channels,
                                                      self.k_height,
                                                      self.k_width))
        self.bias = np.random.uniform(-stdv, stdv, (self.out_channels, 1, 1))

    def print(self, color=""):
        msg = "\tConvolution feature maps: {}, kernel size: {}".format(
            self.out_channels, self.kernel.shape)
        print_in_color(msg, color)

    def print_kernels(self):
        print(self.kernel)

    def set_Lr(self, lr):
        self.lr = lr
        return

    def convolution(self, x, kernel):
        N = x.shape[0]
        in_channel = x.shape[1]
        x_height = x.shape[2]
        x_width = x.shape[3]

        out_channel = kernel.shape[0]
        in_channel = kernel.shape[1]
        k_height = kernel.shape[2]
        k_width = kernel.shape[3]
        stride = 1

        patches = np.asarray([x[n, c, stride*j:stride*j+k_height,
                                stride*k:stride*k+k_width]
                              for n in range(N)
                              for c in range(in_channel)
                              for j in range(x_height-k_height+1)
                              for k in range(x_width-k_width+1)])

        patches = patches.reshape([N, in_channel,
                                   (x_height-k_height+1)*(x_width-k_width+1),
                                   k_height*k_width])

        kernel_repeat = np.repeat(kernel.reshape([out_channel, in_channel, 1,
                                                  k_height*k_width]),
                                  patches.shape[2], axis=2)

        result = np.asarray([np.matmul(kernel_repeat[o, c, j, :],
                                       patches[n, c, j, :])
                             for n in range(N)
                             for o in range(out_channel)
                             for c in range(patches.shape[1])
                             for j in range(patches.shape[2])])

        result = result.reshape([N, kernel_repeat.shape[0],
                                 kernel_repeat.shape[1],
                                 x_height-k_height+1, x_width-k_width+1])
        y = np.sum(result, axis=2)
        y = np.array([y[n,:,:,:] + self.bias for n in range(y.shape[0])])
        return y

    def forward(self, x):
        self.prev_x = x
        self.x_width = x.shape[1]
        self.x_height = x.shape[2]
        y = self.convolution(x, self.kernel)
        return y

    def update(self, dout):
        N, F, W, H = dout.shape
        mean_dout = np.mean(dout, axis=0, keepdims=True)
        mean_x = np.mean(self.prev_x, axis=0, keepdims=True)
        mean_x = np.repeat(mean_x, F, axis=1)

        dk = self.convolution(mean_x, mean_dout)
        dk = np.repeat(dk, self.kernel.shape[1], axis=1)
        self.kernel = self.kernel - self.lr*dk

        db = np.zeros_like(self.bias)
        db = np.sum(dout, axis=(0, 2, 3))
        self.bias = self.bias - self.lr*db.reshape(self.bias.shape)

    def backward(self, dout):
        self.update(dout)

        k_reshaped = np.zeros_like(self.kernel)
        for i in range(self.kernel.shape[-2]):
            for j in range(self.kernel.shape[-1]):
                k_reshaped[:, :, j, i] = np.flip(self.kernel[:, :, i, j])
        k_reshaped = k_reshaped.reshape([self.in_channels,
                                         self.out_channels,
                                         self.k_height,
                                         self.k_width])

        npad = ((0, 0), (0, 0), (self.k_height-1, self.k_height-1),
                (self.k_width-1, self.k_width-1))
        dout = np.pad(dout, pad_width=npad, mode='constant', constant_values=0)

        dy = self.convolution(dout, k_reshaped)
        return dy

    def set_Lr(self, lr):
        self.lr = lr
        return

    def getParametersCount(self):
        return np.prod(self.kernel.shape) + np.prod(self.bias.shape)

    def save(self, path, i):
        with open(path + self.type + i + '-weights.bin', "wb") as f:
            self.kernel.tofile(f)
        with open(path + self.type + i + '-bias.bin', "wb") as f:
            self.bias.tofile(f)

    def load(self, path, i):
        with open(path + self.type + i + '-weights.bin', "rb") as f:
            self.kernel = np.fromfile(f).reshape([self.out_channels,
                                                  self.in_channels,
                                                  self.k_height,
                                                  self.k_width])
        with open(path + self.type + i + '-bias.bin', "rb") as f:
            self.bias = np.fromfile(f).reshape([self.out_channels, 1, 1])


class MaxPooling2D(Module):
    """Max Pooling 2D layer implementation

    Public methods:
    __init__ -- init patch size to apply
    forward -- apply max pooling with 2D patches, save max indexes found
    backward -- reshape data and remap IN values to max indexes
    print -- print class description
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.type = "MaxPooling2D"
        self.kernel_size = kernel_size
        self.stride = kernel_size

    def forward(self, x):
        self.x_shape_origin = x.shape
        x_height = x.shape[2]
        x_width = x.shape[3]

        npad = ((0, 0), (0, 0),
                (0, x_height % self.kernel_size),
                (0, x_width % self.kernel_size))
        x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

        x_n = x.shape[0]
        x_depth = x.shape[1]
        x_height = x.shape[2]
        x_width = x.shape[3]
        self.x_max_idx = np.zeros(self.x_shape_origin)
        y = np.zeros([x_n, x_depth,
                      int(x_height/self.stride), int(x_width/self.stride)])

        for n in range(x_n):
            for c in range(x_depth):
                for j in range(x_height//self.stride):
                    for k in range(x_width//self.stride):
                        i_max = np.argmax(
                            x[n, c,
                              self.stride*j:self.stride*j+self.kernel_size,
                              self.stride*k:self.stride*k+self.kernel_size])
                        idx = [int(i_max > 1) + self.stride*j,
                               int(i_max == 1 or i_max == 3) + self.stride*k]
                        self.x_max_idx[n, c, idx[0], idx[1]] = 1
                        y[n, c, j, k] = x[n, c, idx[0], idx[1]]
        return y

    def backward(self, dout):
        dy = np.zeros(self.x_shape_origin)
        x_n = self.x_shape_origin[0]
        x_depth = self.x_shape_origin[1]
        x_height = self.x_shape_origin[2]
        x_width = self.x_shape_origin[3]

        for n in range(x_n):
            for c in range(x_depth):
                for j in range(int(x_height)):
                    for k in range(int(x_width)):
                        dy[n, c, j, k] = self.x_max_idx[n, c, j, k] *\
                            dout[n, c, j//self.stride, k//self.stride]
        return dy

    def print(self, color=""):
        print_in_color("\tMax Pooling layer, size: " + str(self.kernel_size), color)


class AveragePooling2D(Module):
    """Max Pooling 2D layer implementation

    Public methods:
    __init__ -- init patch size to apply
    forward -- apply mean pooling with 2D patches
    backward -- reshape data, remap IN values to patch location, divide by 4
    print -- print class description
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.type = "AveragePooling2D"
        self.kernel_size = kernel_size
        self.stride = kernel_size

    def forward(self, x):
        self.x_shape_origin = x.shape
        x_height = x.shape[2]
        x_width = x.shape[3]

        npad = ((0, 0), (0, 0),
                (0, x_height % self.kernel_size),
                (0, x_width % self.kernel_size))
        x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

        x_n = x.shape[0]
        x_depth = x.shape[1]
        x_height = x.shape[2]
        x_width = x.shape[3]
        y = np.zeros([x_n, x_depth,
                      int(x_height/self.stride), int(x_width/self.stride)])

        for n in range(x_n):
            for c in range(x_depth):
                for j in range(x_height//self.stride):
                    for k in range(x_width//self.stride):
                        y[n, c, j, k] = np.mean(
                            x[n, c,
                              self.stride*j:self.stride*j+self.kernel_size,
                              self.stride*k:self.stride*k+self.kernel_size])
        return y

    def backward(self, dout):
        dy = np.zeros(self.x_shape_origin)
        x_n = self.x_shape_origin[0]
        x_depth = self.x_shape_origin[1]
        x_height = self.x_shape_origin[2]
        x_width = self.x_shape_origin[3]

        for n in range(x_n):
            for c in range(x_depth):
                for j in range(int(x_height)):
                    for k in range(int(x_width)):
                        dy[n, c, j, k] = dout[n, c, j//self.stride,
                                              k//self.stride] / 4
        return dy

    def print(self, color=""):
        print_in_color("\tAverage Pooling layer, size: " + str(self.kernel_size), color)


class Flatten(Module):
    """Flatten layer implementation to go from convolution to linear

    Public methods:
    __init__ -- initiate class attributes
    forward -- [N, Channels, Width, Height] -> [N, Channels * Width * Height]
    backward -- [N, Channels * Width * Height] -> [N, Channels, Width, Height]
    print -- print class description
    """

    def __init__(self):
        super().__init__()
        self.type = "Flatten"

    def forward(self, x):
        self.n = x.shape[0]
        self.channel = x.shape[1]
        self.width = x.shape[2]
        self.height = x.shape[3]
        y = x.reshape([self.n, self.channel*self.width*self.height])
        return y

    def backward(self, x):
        y = x.reshape([self.n, self.channel, self.height, self.width])
        return y

    def print(self, color=""):
        print_in_color("\tFlatten function", color)
        return


# Sequential architecture
class Sequential(Module):
    """Model implementation to add layers in sequence

    Public methods:
    __init__ -- init list of layers and loss to use
    forward -- apply foward pass for each layer in sequence
    backward -- calculate loss and apply backward pass for each layer
    set_Lr -- set learning rate to use for all layers
    getParametersCount -- return the number of parameters of the model
    save -- save all parameters to .bin files
    load -- load model's parameters from previous training to deploy the model
    print -- print class description
    """

    def __init__(self, param, loss):
        super().__init__()
        self.type = "Sequential"
        self.model = param
        self.loss = loss
        self.train_error = []
        self.test_error = []

    def forward(self, x):
        for layer in self.model:
            x = layer.forward(x)
        return x

    def backward(self, y, y_pred):
        loss = self.loss.loss(y, y_pred)
        derivative = self.loss.derivative(y, y_pred)
        for layer in reversed(self.model):
            derivative = layer.backward(derivative)
        return loss

    def print(self, print_color=True):
        possible_colors = print_in_color("-h")
        if len(possible_colors) < len(__types__):
            print('Not enough color available, {} more\
                needed'.format(len(__types__) - len(possible_colors)))
            print_color = False
            legend = ", ".join([__types__[i] for i in
                                range(len(__types__))])
        elif print_color:
            legend = ", ".join([__types__[i] + " in " +
                                possible_colors[i] for i in
                                range(len(__types__))])
        else:
            legend = ""
        print("Model description: " + legend)
        for layer in self.model:
            if print_color:
                layer.print(possible_colors[
                    __types__.index(layer.type)])
            else:
                layer.print()
        if print_color:
            self.loss.print(possible_colors[
                __types__.index(self.loss.type)])
        else:
            self.loss.print()

    def set_Lr(self, lr=0):
        for layer in self.model:
            try:
                layer.set_Lr(lr)
            except Exception as ex:
                continue

    def getParametersCount(self):
        parametersCount = 0
        for layer in reversed(self.model):
            parametersCount = parametersCount + layer.getParametersCount()
        return parametersCount

    def save(self, path):
        for i, obj in enumerate(self.model):
            params = obj.save(path, str(i))

    def load(self, path):
        for i, obj in enumerate(self.model):
            params = obj.load(path, str(i))
