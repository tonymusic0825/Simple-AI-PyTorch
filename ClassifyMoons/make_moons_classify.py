import torch
from torch import nn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_moon_data(n_samples, noise, seed):
    """Creates moon data

    Parameters:
        n_samples: Number of samples
        noise: Noise of data
        seed: Random seed
    
    Return:
        X_train: X values for training set
        X_test: X values for test set
        y_train: y values for training set
        y_test: y values for test set
    """
    
    X_moon, y_moon = make_moons(n_samples, noise=noise, random_state=seed)

    # USED TO VIEW ON PYPLOT
    # plt.scatter(X_moon[:,0], X_moon[:,1], c= y_moon, cmap=plt.cm.RdYlBu)
    # plt.show()

    # Convert data into torch.Tensors split into test (80%) and test (20%) sets
    X_moon = torch.from_numpy(X_moon).type(torch.float)
    y_moon = torch.from_numpy(y_moon).type(torch.float)

    X_train, X_test, y_train, y_test = train_test_split(X_moon, y_moon, train_size=0.8)

    return X_train, X_test, y_train, y_test

def accuracy(y_pred, y_true):
    """Calculates accuracy of model
    
    Parameters:
        y_pred: Predicted y values
        y_true: The true y values
    
    Returns
        correct: The percentage accuracy of model
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    correct = (correct / len(y_pred)) * 100
    return correct

class ClassifyMoons(nn.Module):
    """Neural network that classifies moon data."""
    def __init__(self):
        super().__init__()

        # The input for this neural network is (X, y) coordinates thus 2 nodes for input layer
        # We will use 20 nodes for hidden layers
        # And there are only 2 boundaries of classification thus the output will be 1 node

        self.input_nodes = 2
        self.hidden_nodes = 20
        self.output_nodes = 1

        self.nn = nn.Sequential(
            nn.Linear(self.input_nodes, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.output_nodes)
        )
    
    def forward(self, x):
        return self.nn(x)


def main():

    SAMPLES = 1000
    RANDOM_SEED = 3907
    NOISE = 0.1

    X_train, X_test, y_train, y_test = create_moon_data(SAMPLES, NOISE, RANDOM_SEED)

    # Create all necessary instances for training model
    model = ClassifyMoons().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epochs = 5000

    # Start training
    for epoch in range(epochs):
        model.train() # Put model into training mode

        y_logits = model(X_train.to(device))
        y_pred = torch.round(torch.sigmoid(y_logits))
        acc = accuracy(y_pred.squeeze(), y_train.to(device))
        loss = loss_fn(y_logits.squeeze(), y_train.to(device))

        # Reset optimizer gradient
        optimizer.zero_grad()
        
        # Backpropagation
        loss.backward()

        # Step optimizer
        optimizer.step()

        # Put model into evaluation mode and try test data
        
        with torch.inference_mode():
            test_logits = model(X_test.to(device))
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_acc = accuracy(test_pred.squeeze(), y_test.to(device))
            test_loss = loss_fn(test_logits.squeeze(), y_test.to(device))

            if epoch % 10 == 0:
                print(f"{epoch} : Train Loss & Acc = {loss:.5f} & {acc:.2f} | Test Loss & Acc = {test_loss:.5f} & {test_acc:.2f}")



if __name__ == "__main__":
    main()