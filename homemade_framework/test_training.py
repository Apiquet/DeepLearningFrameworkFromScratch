from framework import *

# generate train and set set
train_features, train_target = generate_disc_set(1000)
test_features, test_target = generate_disc_set(1000)
# set training en model parameters
nb_epochs = 200
batch_size = 10
hidden_size = 50
learning_rate = 0.003
# Build the model
Model = Sequential([Linear(2, hidden_size),
                    LeakyReLU(),
                    Linear(hidden_size, hidden_size),
                    LeakyReLU(),
                    Linear(hidden_size, 2),
                    Softmax()], LossMSE())
# Set the learning rate
Model.set_Lr(learning_rate)
# Print model's parameters
Model.print(print_color=False)
# start training
train_homemade_model(Model, nb_epochs, train_features, train_target,
                     test_features, test_target, batch_size)
