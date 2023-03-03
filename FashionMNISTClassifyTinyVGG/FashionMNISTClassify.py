"""Classifying FashionMNIST dataset using the TinyVGG Architecture"""

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchinfo import summary

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_data(transform= None):
    """Downloads data and turns data into DataLoader instances

    Parameters:
        transform: 
    
    Returns:
        train_dataloader: A dataloader consisting of train data
        test_dataloader: A dataloader consisting of test data
        class_names: List of data class names
    """

    # Download MNIST data
    train_data = datasets.FashionMNIST(
        root = "FashionMNISTClassify",
        train = True,
        transform = ToTensor(),
        download = True
    )

    test_data = datasets.FashionMNIST(
        root = "FasionMNISTClassify",
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

class TinyVGG(nn.Module):
    """TinyVGG CNN Architecture module"""
    def __init__(self, classes):
        """Initializes model
        
        Parameters
            classes: list of class names of FashionMNIST data
        """
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=490, out_features=len(classes))
        )
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.classifier(x) 

        return x

def main():

    # Setup data
    train_dataloader, test_dataloader, class_names = setup_data()

    # Create all neccessary instances for training
    model = TinyVGG(class_names).to(device)
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