# Deep Learning Framework from scratch, only using numpy

## Project

[This article](https://apiquet.com/2020/05/02/neural-network-from-scratch-part-2/) shows how to create a set of non-linearly separable data and how to implement a FCN: activation functions, linear layer, softmax, MSE loss function, training function and how to build a neural network.

[This article](https://apiquet.com/2020/07/18/deep-learning-framework-from-scratch-part-3/) shows the implementation of Convolution, Flatten, Max and Mean Pooling layers. It will also explained how to implement some good features provided by a Deep Learning Framework such as: saving and loading a model to deploy it somewhere, getting its number of parameters, drawing learning curves, printing its description.

## Layer implementation:

* Linear layer
* Convolution layer
* Flatten layer
* Max pooling layer
* Mean pooling layers
* Batch Normalization
* Activation layers: Sigmoid, ReLU, LeakyReLU
* Softmax
* Loss function (MSE: Mean Squared Error)

## Layer to build a neural network model:

* Sequential module to create a neural network. Declaration of a simple CNN model with: Convolution -> LeakyReLU -> Max Pooling -> Convolution -> LeakyReLU -> Flatten -> Batch Normalization -> Linear layer -> Linear layer

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
	NN.LossMSE())
```

## Useful features of the framework:

* One-hot encoding conversion
* Training function (epoch and mini-batch)
* Saving and loading a model to deploy it somewhere
* Getting the number of parameters of a created model
* Drawing learning curves
* Printing model's description

## Example of use

The following example contains an implementation and training of CNN. It also shows the save and load function to deploy the model somewhere.

Finally, it compares with an FCN model: number of parameters / accuracy.

https://github.com/Apiquet/DeepLearningFrameworkFromScratch/blob/master/cnn_example.ipynb
