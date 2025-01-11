import pandas as pd
import numpy as np

# read feature dataset
train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']

valid_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
valid_feat_X = valid_feat['features']
valid_feat_Y = valid_feat['label']

test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']

# many columns have unique value
num_examples, num_vectors, dim_vectors = train_feat_X.shape
useful_cols = [i for i in range(num_vectors) if max([pd.Series(train_feat_X[:,i,j]).nunique() for j in range(dim_vectors)]) > 1]
useful_cols

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import random

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_channels, num_classes=2, kernel_size=2, dropout=0.2):
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        y = self.tcn(x)
        return self.fc(y[:, :, -1])

    def train_model(self, train_loader, criterion, optimizer, num_epochs=10):
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.permute(0, 2, 1)  
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    def evaluate_model(self, dataloader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.permute(0, 2, 1)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return (100 * correct / total)

    
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def print_classification_report_and_confusion_matrix(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in dataloader:
            inputs = inputs.permute(0, 2, 1)  # Adjust input dimensions
            outputs = model(inputs)  # Forward pass through the model
            _, predicted = torch.max(outputs, 1)  # Get predicted class (index of max logit)

            # Store true labels and predicted labels for evaluation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Print classification report
    print("Classification Report:")
    print("----------------------------------------------------------")
    print(classification_report(all_labels, all_predictions))
    # print("----------------------------------------------------------")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
    print("\n\n")
    

    
import random
import numpy as np
import torch

def reset_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    

# Prepare DataLoader
def prepare_dataloader(X, y, batch_size=32):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Function to train on fractions of the data and return the best model
def train_on_fractions(fractions, X, y, valid_loader, input_size, num_channels, batch_size, num_epochs, learning_rate):
    best_acc = 0
    best_frac = 0
    valid_accuracies = []

    n_samples = len(X)
    full_model = None
    for fraction in fractions:
        # Subset of the data
        subset_size = int(n_samples * fraction)
        subset_X = X[:subset_size]
        subset_y = y[:subset_size]

        # Prepare DataLoader
        train_loader = prepare_dataloader(subset_X, subset_y, batch_size)

        best_model = None
        acc = 0.0
        
        # Reset seeds and initialize the model
        for seed in [0,1,2]:
            reset_random_seeds(seed)  # Reset with a seed value
            model = TCNClassifier(input_size=input_size, num_channels=num_channels)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            # Train the model
            model.train_model(train_loader, criterion, optimizer, num_epochs)

            # Evaluate on validation data
            if model.evaluate_model(valid_loader) > acc:
                acc = model.evaluate_model(valid_loader)
                best_model = model
        valid_accuracies.append(acc)
        print("----------------------------------------------------------")
        print(f"Fraction: {fraction}, Validation Accuracy: {acc:.2f}%")
        print("----------------------------------------------------------")
        
        print_classification_report_and_confusion_matrix(best_model, valid_loader) #

        # Track best model
        if acc > best_acc:
            best_acc = acc
            best_frac = fraction
        
        full_model = best_model
        
        

    # Plot the accuracies for each fraction
    plt.plot(fractions, valid_accuracies, marker='o')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy vs. Training Data Fraction')
    plt.show()
    
    print("=====================================================================")
    print(f"Best Validation accuracy was for fraction of {best_frac}: {best_acc:.2f}%")
    print("=====================================================================\n\n")
    
    return full_model



fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

# Hyperparameters
n_vectors = 3
vector_dim = int(768 * 1/2) # int(768 * 3.8 / 5)
n_samples = 7080
num_channels = [4,16] # [4, 32]  # Number of channels for each layer in the TCN
batch_size = 64
num_epochs = 15
learning_rate = 0.0015

# Training and validation data converted to Tensors
X = torch.from_numpy(train_feat_X[:, useful_cols, : vector_dim]).float()
y = torch.from_numpy(train_feat_Y[:]).long()
valid_X = torch.from_numpy(valid_feat_X[:, useful_cols, : vector_dim]).float()
valid_y = torch.from_numpy(valid_feat_Y[:]).long()

# Prepare Validation DataLoader
valid_loader = prepare_dataloader(valid_X, valid_y, batch_size)

full_model = train_on_fractions(fractions, X, y, valid_loader, vector_dim, num_channels, batch_size, num_epochs, learning_rate)




# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install("torchinfo")
# # !pip install torchsummary
# from torchinfo import summary

# summary(full_model)


# Ensure the model is in evaluation mode
full_model.eval()

# Assuming test_X is a tensor of test data, and its shape is (batch_size, sequence_length, input_size)
test_X = torch.from_numpy(test_feat_X[:, useful_cols, : vector_dim]).float()

# We permute test_X to match the expected input shape for the TCN: (batch_size, input_size, sequence_length)
test_X = test_X.permute(0, 2, 1)

# Perform predictions on test_X
with torch.no_grad():
    outputs = full_model(test_X)
    _, predicted = torch.max(outputs, 1)  # Get the predicted class index

# Save predictions to a text file, one prediction per line
with open('pred_deepfeat.txt', 'w') as f:
    for pred in predicted:
        f.write(f"{pred.item()}\n")  # Convert the tensor element to a Python integer and write it to the file

print("Predictions successfully saved to 'pred_deepfeat.txt'")