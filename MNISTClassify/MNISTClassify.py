import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_data():
    """Downloads data and turns data into DataLoader instances
    
    Returns:
        train_dataloader: A dataloader consisting of train data
        test_dataloader: A dataloader consisting of test data
        class_names: List of data class names
    """

    # Download MNIST data
    train_data = datasets.MNIST(
        root = "MNISTClassify",
        train = True,
        transform = ToTensor(),
        download = True
    )

    test_data = datasets.MNIST(
        root = "MNISTClassify",
        train = False,
        transform = ToTensor(),
        download = True
    )

    class_names = train_data.classes
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)

    return train_dataloader, test_dataloader, class_names

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

class MNISTClassify(nn.Module):
    """Neural network that classifies MNIST (Handwritten numbers) data."""

    def __init__(self, classes):
        """Initializes model
        
        Parameters:
            classes: list of classes of dataset
        """
        super().__init__()

        self.input_dim = 28**2 # Since our photo is 28 x 28 pixels
        self.hidden_nodes = 15
        self.output_nodes = len(classes)

        # We must flatten (32, 1, 28, 28) -> (32, 784) 
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.nn = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.output_nodes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.nn(x) 

def main():

    # Create dataset
    train_dataloader, test_dataloader, class_names = setup_data()

    # Create all neccessary instances for training
    model = MNISTClassify(class_names).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 20

    # Train model
    for epoch in range(epochs):
        train_acc = 0 # For average train accuracy
        for batch, (X, y) in enumerate(train_dataloader):
            model.train() # Put model into training mode

            y_logits = model(X.to(device))
            loss = loss_fn(y_logits, y.to(device))
            y_pred = y_logits.argmax(dim=1)
            train_acc += accuracy(y_pred, y.to(device))

            # Reset optimizer grad
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Step optimizer
            optimizer.step()

            # Put model into evaluation mode and try test data
            model.eval()

        with torch.inference_mode():
            test_acc = 0
            for batch, (X_test, y_test) in enumerate(test_dataloader):
                test_logits = model(X_test.to(device))
                test_loss = loss_fn(test_logits, y_test.to(device))
                test_pred = test_logits.argmax(dim=1)
                test_acc += accuracy(test_pred, y_test.to(device))
            
            # Calculate average accuracy of model on test data
            test_acc /= len(test_dataloader)
            train_acc /= len(train_dataloader)
        
        print(f"{epoch} : Train Loss & Acc = {loss:.5f} & {train_acc:.2f} | Test Loss & Acc = {test_loss:.5f} & {test_acc:.2f}")

        

if __name__ == "__main__":
    main()

        

