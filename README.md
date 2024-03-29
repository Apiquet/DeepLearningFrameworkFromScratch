# Deep Learning Framework from scratch, only using numpy

## Project

**Gradient Descent**: [This article](https://apiquet.com/2020/03/28/neural-net-from-scratch-part-1/) describes how to implement a gradient descent using the differential approach (2D example implementation), then, using the perturbation approach (3D example implementation).

**FCN implementation**: [This article](https://apiquet.com/2020/05/02/neural-network-from-scratch-part-2/) shows how to create a set of non-linearly separable data and how to implement a FCN from scratch using numpy: linear layers, activation functions, loss and training function.

**Python Deep Learning Framework implementation**: [This article](https://apiquet.com/2020/07/18/deep-learning-framework-from-scratch-part-3/) shows the implementation of a Deep Learning Framework with only numpy. It implements all the layers listed in the next section. It also explains how to implement some good features provided by a Deep Learning Framework such as: saving and loading a model to deploy it somewhere, getting its number of parameters, drawing learning curves, printing its description, getting its confusion matrix, etc.

**Network deployment**: [This article](https://apiquet.com/2020/08/21/neural-network-from-scratch-part-4/) explains how the Deep Learning Framework can help to create and train a CNN for hand signal recognition for UAV piloting. It also shows how to build a dataset for a particular task and how to deploy a trained model to perform the task.

**C++ Deep Learning Framework implementation**: [This article](https://apiquet.com/2021/12/12/neural-network-from-scratch-part-5-c-deep-learning-framework-implementation/) explains how to create a C++ library that implements a simple Deep Learning Framework: Linear layer, MSE loss, ReLU and Softmax functions, a feature/label generator and a training loop. The main goal of this article is to show how to develop a project in C++ by explaining key concepts of the language.

## Layer implementation:

* Linear layer
* Convolution layer
* Flatten layer
* Max pooling layer
* Average pooling layer
* Batch Normalization
* Activation functions: Sigmoid, ReLU, LeakyReLU, Softmax
* Loss functions (MSE: Mean Squared Error, Cross Entropy)

## Example to build a neural network model:

* To build a model, the Sequential module has to be used. The code below creates the following network: Convolution -> LeakyReLU -> Max Pooling -> Convolution -> LeakyReLU -> Flatten -> Batch Normalization -> Linear layer -> Linear layer -> Softmax -> Cross Entropy Loss

```
cnn_model = NN.Sequential([
	NN.Convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
	NN.LeakyReLU(),
	NN.MaxPooling2D(2),
	NN.Convolution(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size),
	NN.LeakyReLU(),
	NN.Flatten(),
	NN.BatchNorm(),
	NN.Linear((out_first_conv**2)*out_channels, hidden_size),
	NN.LeakyReLU(),
	NN.BatchNorm(),
	NN.Linear(hidden_size, num_class),
	NN.Softmax()],
	NN.LossCrossEntropy())
```

## Useful features of the framework:

* One-hot encoding conversion
* Train function (epoch and mini-batch)
* Save and load a model to deploy it somewhere
* Get the number of parameters of a created model
* Get the confusion matrix
* Draw learning curves
* Print model's description

## Example of use

The cnn-fcn_example.ipynb notebooks contain an implementation and training of a CNN model and a FCN model. They also show how to save and load a model with its weights to deploy it somewhere.

The homemadeframework_vs_pytorch.ipynb notebook shows a comparison with pytorch framework.

