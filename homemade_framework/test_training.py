#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to test homemade neural network

Args:
    - (OPTIONAL) -e number of epochs (default is 200)
    - (OPTIONAL) -b mini-batch size (default is 10)
    - (OPTIONAL) -s hidden size, number of neurons per layers (default is 50)
    - (OPTIONAL) -l learning rate (default is 0.003)
    - (OPTIONAL) -n number of samples for training (default is 1000)
    - (OPTIONAL) -t number of samples for testing (default is 1000)

Example:
    python3 test_training.py
"""

import argparse
from framework import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--num_epochs",
        required=False,
        default=200,
        type=int,
        help="Number of epochs (default is 200)."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        required=False,
        default=10,
        type=int,
        help="Mini-batch size (default is 10)."
    )
    parser.add_argument(
        "-s",
        "--hidden_size",
        required=False,
        default=50,
        type=int,
        help="Hidden size: number of neurons for the layers (default is 50)."
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        required=False,
        default=0.003,
        type=float,
        help="Learning rate."
    )
    parser.add_argument(
        "-n",
        "--num_samples_train",
        required=False,
        default=1000,
        type=int,
        help="Number of samples to generate for training (default is 1000)."
    )
    parser.add_argument(
        "-t",
        "--num_samples_test",
        required=False,
        default=1000,
        type=int,
        help="Number of samples to generate for testing (default is 1000)."
    )

    args = parser.parse_args()

    # generate train and set set
    train_features, train_target = generate_disc_set(args.num_samples_train)
    test_features, test_target = generate_disc_set(args.num_samples_test)

    # Build the model
    Model = Sequential([Linear(2, args.hidden_size),
                        LeakyReLU(),
                        Linear(args.hidden_size, args.hidden_size),
                        LeakyReLU(),
                        Linear(args.hidden_size, 2),
                        Softmax()], LossMSE())

    # Set the learning rate
    Model.set_Lr(args.learning_rate)

    # Print model's parameters
    Model.print(print_color=False)

    # start training
    train_homemade_model(Model, args.num_epochs, train_features, train_target,
                         test_features, test_target, args.batch_size)


if __name__ == '__main__':
    main()
