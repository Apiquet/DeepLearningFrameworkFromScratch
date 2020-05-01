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

# Convert train_target to one hot encoding
train_target_one_hot = convert_to_one_hot_labels(train_features, train_target)

print_current_results(0, Model, train_features, train_target, test_features, test_target, 0, prefix = "Before training: ")
test_results = []
for epochs in range(0, nb_epochs):
    loss_sum = 0
    test_results.append(get_inferences(Model, test_features))
    for b in range(train_features.shape[0] // batch_size):
        output = Model.forward(train_features[list(range(b * batch_size, (b+1) * batch_size))])
        loss = Model.backward(train_target_one_hot[list(range(b * batch_size, (b+1) * batch_size))], output)
        loss_sum = loss_sum + loss.item()
    if epochs % 30 == 0:
        print_current_results(epochs + 1, Model, train_features, train_target, test_features, test_target, loss_sum)
    
        
print_current_results(epochs, Model, train_features, train_target, test_features, test_target, loss_sum, prefix = "After training: ")

    