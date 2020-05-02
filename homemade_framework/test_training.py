from framework import *

train_features, train_target = generate_disc_set(1000)
test_features, test_target = generate_disc_set(1000)

nb_epochs = 400
batch_size = 10
hidden_size = 50

# Build the model
Model = Sequential([Linear(2, hidden_size),
                    LeakyReLU(),
                    Linear(hidden_size, hidden_size),
                    LeakyReLU(),
                    Linear(hidden_size, 2),
                    Softmax()], LossMSE())
# Set the learning rate
Model.set_Lr(0.003)

# Print model's parameters
Model.print(print_color=False)

train_homemade_model(Model, nb_epochs, train_features, train_target,
                     test_features, test_target, batch_size)
