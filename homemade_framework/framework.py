import matplotlib.pyplot as plt
import numpy as np
import math


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
               "magenta": "35", "red": "31"}
    if message == "-h":
        return list(choices.keys())
    elif color == "":
        print(message)
    elif color in choices:
        print("\x1b[" + choices[color] + "m" + message + "\x1b[0m")
    else:
        raise ValueError("Available colors: {}, '-h' to get\
            the list".format(choices.keys()))


def train_homemade_model(model, num_epochs, train_features_np,
                         train_target_np, test_features_np,
                         test_target_np, batch_size):
    start_time = datetime.datetime.now()
    # Convert train_target to one hot encoding
    train_target_one_hot = convert_to_one_hot_labels(train_features_np,
                                                     train_target_np)

    print_current_results(0, model, train_features_np, train_target_np,
                          test_features_np, test_target_np, 0,
                          prefix="Before training: ")
    test_results = []
    for epochs in range(0, num_epochs):
        loss_sum = 0
        test_results.append(NN.get_inferences(Model, test_features_np))
        for b in range(train_features.shape[0] // batch_size):
            output = model_.forward(train_features_np[
                list(range(b*batch_size, (b+1)*batch_size))])
            loss = model.backward(train_target_one_hot[
                list(range(b*batch_size, (b+1)*batch_size))],
                                  output)
            loss_sum = loss_sum + loss.item()
        if epochs % 30 == 0:
            print_current_results(epochs + 1, model, train_features_np,
                                  train_target_np, test_features_np,
                                  test_target_np, loss_sum)

    training_time = datetime.datetime.now() - start_time
    print('\nTraining time: {}'.format(training_time))
    print_current_results(epochs, Model, train_features_np, train_target_np,
                          test_features_np, test_target_np, loss_sum,
                          prefix="After training: ")


# Data Manager
def generate_disc_set(nb):
    features = np.random.uniform(-1, 1, (nb, 2))
    target = []
    for el in features:
        target.append(int(((math.pow(el[0], 2) +
                            math.pow(el[1], 2))/math.pi) < 0.2))
    return features, np.asarray(target)


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
possible_types = ["Linear", "Activation", "Loss", "Softmax"]


# heritage module definition
class Module(object):
    def __init__(self):
        super().__init__()
        self.lr = 0

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError


# RelU activation function
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.type = "Activation"
        self.save = 0

    def forward(self, x):
        self.save = x
        x[x < 0] = 0
        y = x
        return y

    def backward(self, x):
        y = (self.save > 0).astype(float)
        return np.multiply(y, x)

    def print(self, color=""):
        print_in_color("\tReLU activation", color)
        return


# LeakyReLU activation function
class LeakyReLU(Module):
    def __init__(self):
        super().__init__()
        self.type = "Activation"
        self.save = 0
        self.a = 0.01

    def forward(self, x):
        self.save = x
        neg = x < 0
        pos = x >= 0
        y = np.multiply(neg.astype(float), x)*self.a +\
            np.multiply(pos.astype(float), x)
        return y

    def backward(self, x):
        neg = self.save < 0
        pos = self.save >= 0
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
        self.save = 0

    def eq(self, x):
        return 1 / (1 + np.exp(np.multiply(x, -1)))

    def forward(self, x):
        self.save = x
        y = self.eq(x)
        return y

    def backward(self, x):
        y = np.multiply(self.eq(self.save) * (1 - self.eq(self.save)), x)
        return y

    def print(self, color=""):
        print_in_color("\tSigmoid activation", color)
        return


# MSE Loss implementation
class LossMSE(Module):
    def __init__(self):
        super().__init__()
        self.type = "Loss"

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
        self.save = 0

    def eq(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1)[:, None]

    def forward(self, x):
        self.save = x
        y = self.eq(x)
        return y

    def backward(self, x):
        y = np.multiply(self.eq(self.save) * (1 - self.eq(self.save)), x)
        return y

    def print(self, color=""):
        print_in_color("\tSoftmax function", color)
        return


# Linear layer
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.type = "Linear"
        self.x = np.zeros(out_features)
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
            np.multiply(lr, np.matmul(np.transpose(self.x), grad))
        self.bias = self.bias -\
            lr*grad.mean(0).reshape([self.bias.shape[0], 1])*1

    def backward(self, grad):
        b = np.matmul(grad, np.transpose(self.weight))
        self.update(grad)
        return b

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.weight) +\
            np.transpose(np.repeat(self.bias, x.shape[0], axis=1))

    def set_Lr(self, lr):
        self.lr = lr
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
            if isinstance(_object, Linear):
                _object.set_Lr(lr)
