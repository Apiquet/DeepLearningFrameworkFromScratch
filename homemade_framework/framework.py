from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np


# Utils
def print_current_results(epochs, Model, train_features, train_target,
                          test_features, test_target, loss_sum, prefix=""):
    train_error = compute_accuracy(Model, train_features, train_target)
    test_error = compute_accuracy(Model, test_features, test_target)
    print(prefix + "Epoch: {}, Train Error: {:.4f}%,\
        Test Error: {:.4f}%, Loss  {:.4f}".format(epochs, train_error,
                                                  test_error, loss_sum))


def print_in_color(message, color="red"):
    choices = {"green": "32", "blue": "34",
               "magenta": "35", "red": "31",
               "Gray": "37", "Cyan": "36",
               "Black": "39"}
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
                         test_target, batch_size):
    start_time = datetime.now()
    # Convert train_target to one hot encoding
    train_target_one_hot = convert_to_one_hot_labels(train_features,
                                                     train_target)

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
        if epochs % 30 == 0:
            print_current_results(epochs + 1, model, train_features,
                                  train_target, test_features,
                                  test_target, loss_sum)

    training_time = datetime.now() - start_time
    print('\nTraining time: {}'.format(training_time))
    print_current_results(epochs, model, train_features, train_target,
                          test_features, test_target, loss_sum,
                          prefix="After training: ")


# Data Manager
def generate_disc_set(nb):
    features = np.random.uniform(-1, 1, (nb, 2))
    target = (features[:, 0]**2 + features[:, 1]**2)/math.pi < 0.2
    return features, target.astype(int)


def plot_dataset(features, target):
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.title("Dataset")
    scatter = ax.scatter(features[:, 0], features[:, 1],
                         c=target)
    legend = ax.legend(*scatter.legend_elements(), title="Labels",
                       loc="lower right")
    ax.add_artist(legend)
    return plt


def convert_to_one_hot_labels(features, target):
    n_values = max(target) + 1
    target_onehot = np.eye(n_values)[target]
    return target_onehot


def compute_accuracy(model, data_features, data_target):
    predicted_classes = get_inferences(model, data_features)
    nb_data_errors = sum(data_target != predicted_classes)
    return nb_data_errors/data_features.shape[0]*100


def get_inferences(model, data_features):
    output = model.forward(data_features)
    predicted_classes = np.argmax(output, axis=1)
    return predicted_classes


# Classes
possible_types = ["Linear", "Activation", "Loss",
                  "Softmax", "Flatten", "Convolution",
                  "Batch_normalization"]


# heritage module definition
class Module(object):
    def __init__(self):
        super().__init__()
        self.lr = 0

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def save(self, path, i):
        return

    def load(self, path, i):
        return


# RelU activation function
class ReLU(Module):
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

    def backward(self, x):
        y = (self.prev_x > 0).astype(float)
        return np.multiply(y, x)

    def print(self, color=""):
        print_in_color("\tReLU activation", color)
        return


# LeakyReLU activation function
class LeakyReLU(Module):
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

    def backward(self, x):
        neg = self.prev_x < 0
        pos = self.prev_x >= 0
        y = np.multiply(neg.astype(float), x)*self.a +\
            np.multiply(pos.astype(float), x)
        return y

    def print(self, color=""):
        print_in_color("\tLeakyReLU activation", color)
        return


# sigmoid activation function
class Sigmoid(Module):
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


# MSE Loss implementation
class LossMSE(Module):
    def __init__(self):
        super().__init__()
        self.type = "Loss"
        self.name = "LossMSE"

    def loss(self, y, y_pred):
        loss = sum(((y_pred - y)**2).sum(axis=0))/y.shape[1]
        return loss

    def print(self, color=""):
        print_in_color("\tMSE", color)

    def grad(self, y, y_pred):
        return 2*(y_pred-y)/y.shape[1]


# Softmax function implementation
class Softmax(Module):
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


# Batch normalization function implementation
class Batch_normalization(Module):
    def __init__(self):
        super().__init__()
        self.type = "Batch_normalization"
        self.gamma = 1
        self.eps = 10**-100
        self.beta = 0

    def eq(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1)[:, None]

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
        self.gamma = self.gamma - self.lr * dgamma
        self.beta = self.beta - self.lr * dbeta

    def set_Lr(self, lr):
        self.lr = lr
        return

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
        print_in_color("\tBatch normalization function", color)
        return


# Linear layer
class Linear(Module):
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

    def print(self, color=""):
        msg = "\tLinear layer shape: {}".format([self.weight.shape[0],
                                                 self.weight.shape[1]])
        print_in_color(msg, color)

    def print_weight(self):
        print(self.weight)

    def update(self, grad):
        lr = self.lr
        self.weight = self.weight -\
            np.multiply(lr, np.matmul(np.transpose(self.prev_x), grad))
        self.bias = self.bias -\
            lr*grad.mean(0).reshape([self.bias.shape[0], 1])*1

    def backward(self, grad):
        b = np.matmul(grad, np.transpose(self.weight))
        self.update(grad)
        return b

    def forward(self, x):
        self.prev_x = x
        return np.matmul(x, self.weight) +\
            np.transpose(np.repeat(self.bias, x.shape[0], axis=1))

    def set_Lr(self, lr):
        self.lr = lr
        return

    def save(self, path, i):
        print(i, self.weight.shape)
        with open(path + self.type + i + '-weights.bin', "wb") as f:
            self.weight.tofile(f)
        with open(path + self.type + i + '-bias.bin', "wb") as f:
            self.bias.tofile(f)
        return [self.type, self.weight, self.bias]

    def load(self, path, i):
        with open(path + self.type + i + '-weights.bin', "rb") as f:
            self.weight = np.fromfile(f).reshape([self.in_features, self.out_features])
        print(i, self.weight.shape)
        with open(path + self.type + i + '-bias.bin', "rb") as f:
            self.bias = np.fromfile(f).reshape([self.out_features, 1])


# Convolutional layer
class Convolution(Module):
    def __init__(self, in_channels=1, out_channels=5,
                 kernel_size=3, stride=1, padding=1):
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

        # print("x shape", x.shape)
        # print("kernel shape", kernel.shape)

        patches = np.asarray([x[n, c, stride*j:stride*j+k_height,
                                stride*k:stride*k+k_width]
                              for n in range(N)
                              for c in range(in_channel)
                              for j in range(x_height-k_height+1)
                              for k in range(x_width-k_width+1)])
        # print("patches shape", patches.shape)
        patches = patches.reshape([N, in_channel,
                                   (x_height-k_height+1)*(x_width-k_width+1),
                                   k_height*k_width])
        # print("patches shape", patches.shape)
        kernel_repeat = np.repeat(kernel.reshape([out_channel, in_channel, 1,
                                                  k_height*k_width]),
                                  patches.shape[2], axis=2)
        # print("kernel_repeat shape", kernel_repeat.shape)
        result = np.asarray([np.matmul(kernel_repeat[o, c, j, :],
                                       patches[n, c, j, :])
                             for n in range(N)
                             for o in range(out_channel)
                             for c in range(patches.shape[1])
                             for j in range(patches.shape[2])])
        # print("result shape", result.shape)
        result = result.reshape([N, kernel_repeat.shape[0],
                                 kernel_repeat.shape[1],
                                 x_height-k_height+1, x_width-k_width+1])
        y = np.sum(result, axis=2)
        # print("y shape", y.shape)
        return y

    def update(self, grad):
        dk = self.convolution(self.prev_x, grad)
        self.kernel = self.kernel - self.lr*dk

    def forward(self, x):
        self.prev_x = x
        self.x_width = x.shape[1]
        self.x_height = x.shape[2]
        y = self.convolution(x, self.kernel)
        # print("conv forward, k shape", self.kernel.shape)
        # print("conv forward, y shape", y.shape)
        return y

    def backward(self, grad):
        self.update(grad)
        k_reshaped = np.ones([kernel.shape[1], kernel.shape[0],
                              kernel.shape[2], kernel.shape[3]])
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                k_reshaped[j, i, :, :] = np.flip(kernel[i, j, :, :])
        # todo add kernel reshape
        padding = (grad.shape[0], self.k_height-1, self.k_width-1)
        dout = np.array([np.pad(grad[i, :, :], [padding, padding],
                         mode='constant', constant_values=0)
                         for i in range(grad.shape[0])])

        dy = self.convolution(dout, k_reshaped)
        return dy

    def set_Lr(self, lr):
        self.lr = lr
        return


# Flatten function implementation
class Flatten(Module):
    def __init__(self):
        super().__init__()
        self.type = "Flatten"

    def forward(self, x):
        # print("flatten forward, x shape", x.shape)
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
    def __init__(self, param, loss):
        super().__init__()
        self.type = "Sequential"
        self.model = param
        self.loss = loss

    def forward(self, x):
        for _object in self.model:
            x = _object.forward(x)
        return x

    def backward(self, y, y_pred):
        loss = self.loss.loss(y, y_pred)
        grad_pred = self.loss.grad(y, y_pred)
        for _object in reversed(self.model):
            grad_pred = _object.backward(grad_pred)
        return loss

    def print(self, print_color=True):
        possible_colors = print_in_color("-h")
        if len(possible_colors) < len(possible_types):
            print('Not enough color available, {} more\
                needed'.format(len(possible_types) - len(possible_colors)))
            print_color = False
        elif print_color:
            legend = ", ".join([possible_types[i] + " in " +
                                possible_colors[i] for i in
                                range(len(possible_types))])
        else:
            legend = ""
        print("Model description: " + legend)
        for _object in self.model:
            if print_color:
                _object.print(possible_colors[
                    possible_types.index(_object.type)])
            else:
                _object.print()
        if print_color:
            self.loss.print(possible_colors[
                possible_types.index(self.loss.type)])
        else:
            self.loss.print()

    def set_Lr(self, lr=0):
        for _object in self.model:
            try:
                _object.set_Lr(lr)
            except Exception as ex:
                continue

    def save(self, path):
        for i, obj in enumerate(self.model):
            params = obj.save(path, str(i))

    def load(self, path):
        for i, obj in enumerate(self.model):
            params = obj.load(path, str(i))
