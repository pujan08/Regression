import torch
from torch import nn


def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    """
    model = nn.Linear(input_size, output_size)
    return model


def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    """
    learning_rate = 0.001  # Adjusted learning rate
    num_epochs = 10000  # Number of epochs
    input_features = X.shape[1]  # Extract the number of features from the input
    output_features = y.shape[1]  # Extract the number of features from the output
    model = create_linear_regression_model(input_features, output_features)

    loss_fn = nn.MSELoss()  # Use mean squared error loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    previous_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        # Print the loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        # Stop training if the loss is not changing much
        if abs(previous_loss - loss.item()) < 1e-6:
            print(f"Converged at epoch {epoch}")
            break
        previous_loss = loss.item()
    return model, loss