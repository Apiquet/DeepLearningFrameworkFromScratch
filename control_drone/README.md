# How to control a drone with hand signs

## Project

The goal of this project is to use the Deep Learning Framework implemented with numpy to build a neural network capable of controling an Anafi drone.
To do so, a dataset how hand sign should be created with the script /DB/db_creation.py
Then, the neural network will be trained on it to learn the classification of each sign.
An option is also available to use a TensorFlow model instead of the framework implemented in this repository.

Once the model is trained, we can turn on the Anafi drone, enable wifi connection with the computer and run: run_model_on_cam.py with appropriate arguments.

## DB creation, training and drone control:

Complete tutorial [here](https://apiquet.com/2020/08/21/neural-network-from-scratch-part-4/).
