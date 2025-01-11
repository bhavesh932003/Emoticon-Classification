import pandas as pd
import numpy as np

# read text sequence dataset
train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
train_seq_X = train_seq_df['input_str'].tolist()
train_seq_Y = train_seq_df['label'].tolist()

valid_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv")
valid_seq_X = valid_seq_df['input_str'].tolist()
valid_seq_Y = valid_seq_df['label'].tolist()

test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define the LSTM model with learned embeddings
class LSTMClassifierWithEmbedding(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, learning_rate=0.001):
        super(LSTMClassifierWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

    # Method to train the model
    def train_model(self, train_loader, num_epochs):
        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_sequences, batch_labels in train_loader:
                outputs = self(batch_sequences)
                loss = self.criterion(outputs, batch_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            # Optionally print the epoch loss
            # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}')
        return self

    # Method to evaluate the model
    def evaluate_model(self, val_loader):
        self.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for val_sequences, val_labels in val_loader:
                outputs = self(val_sequences)
                predictions = torch.sigmoid(outputs).round()
                all_labels.extend(val_labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy

    
    
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def print_classification_report_and_confusion_matrix(model, dataloader, class_names=None):
    """
    Prints the classification report and confusion matrix for a given PyTorch model and dataloader.

    Parameters:
    model (nn.Module): Trained PyTorch model.
    dataloader (DataLoader): Dataloader containing validation/test data.
    class_names (list, optional): List of class names for labeling confusion matrix. Defaults to None.
    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs).round()  # Assuming binary classification, modify for multi-class
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    # Generate the classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    
    
    
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
    
    
    
    
# Main function to encapsulate the entire training, validation, and best model saving process
def run_training(fractions, sequence_tensors, labels_tensor, val_sequence_tensors, val_labels_tensor, input_size, embedding_dim, hidden_size, output_size, learning_rate, num_epochs, batch_size):
    val_accuracies = []
    full_model = None
    best_accuracy = 0.0
    best_frac = 0.0

    val_dataset = TensorDataset(val_sequence_tensors, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for fraction in fractions:
        num_samples = int(len(sequence_tensors) * fraction)
        fraction_sequences = sequence_tensors[:num_samples]
        fraction_labels = labels_tensor[:num_samples]

        train_dataset = TensorDataset(fraction_sequences, fraction_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_model = None
        val_accuracy = 0.0
        # Reset seeds and initialize the model
        for seed in [0,1]:
            reset_random_seeds(seed)  # Reset with a seed value
            model = LSTMClassifierWithEmbedding(input_size, embedding_dim, hidden_size, output_size, learning_rate)
            # Train the model
            model.train_model(train_loader, num_epochs)
            # Evaluate the model on the validation set
            if model.evaluate_model(val_loader) > val_accuracy:
                val_accuracy = model.evaluate_model(val_loader)
                best_model = model
        val_accuracies.append(val_accuracy * 100)
        
        print("----------------------------------------------------------")
        print(f"Fraction: {fraction}, Validation Accuracy: {val_accuracy * 100:.2f}%")
        print("----------------------------------------------------------")
        
        class_names = ['Class 0', 'Class 1']
        print_classification_report_and_confusion_matrix(best_model, val_loader, class_names)

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_frac = fraction
        
        full_model = best_model
        
    # Plot validation accuracy vs. fraction of training data
    plt.plot(fractions, val_accuracies, marker='o')
    plt.title('Validation Accuracy vs. Fraction of Training Data')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Validation Accuracy')
    plt.xticks(fractions)
    plt.grid()
    plt.show()

    # Save the best model to disk
    # torch.save(best_model, 'best_lstm_model.pth')
    print("=======================================")
    print(f'Best Validation Accuracy was for Fraction = {best_frac}: {best_accuracy*100:.2f}%')
    print("=======================================")
    
    return best_model




# Convert sequence of digits into tensor (just a list of numbers)
def digit_sequence_to_tensor(sequence):
    return torch.tensor([int(digit) for digit in sequence], dtype=torch.long)



# Convert sequences to tensor format (each sequence is a list of digit indices)
sequence_tensors = [digit_sequence_to_tensor(seq) for seq in train_seq_X]
sequence_tensors = torch.stack(sequence_tensors)  # Stack to form a batch tensor

val_sequence_tensors = [digit_sequence_to_tensor(seq) for seq in valid_seq_X]
val_sequence_tensors = torch.stack(val_sequence_tensors)  # Stack to form a batch tensor

# Convert labels to tensor
labels_tensor = torch.tensor(train_seq_Y).float().unsqueeze(1)  # Convert labels to tensor and reshape

val_labels_tensor = torch.tensor(valid_seq_Y).float().unsqueeze(1)




# Hyperparameters
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
input_size = 10        # Number of possible digits (0-9)
embedding_dim = 16     # Size of the embedding vector for each digit
hidden_size = 40       # Size of the LSTM hidden state
output_size = 1        # Binary classification (0 or 1)
learning_rate = 0.0015
num_epochs = 25 # 18 # 25
batch_size = 16 # 8 # 16

# Prepare your data as tensors (sequence_tensors, labels_tensor, val_sequence_tensors, val_labels_tensor)

# Call the run_training function
full_model = run_training(fractions, sequence_tensors, labels_tensor, val_sequence_tensors, val_labels_tensor, input_size, embedding_dim, hidden_size, output_size, learning_rate, num_epochs, batch_size)



# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install("torchinfo")

# from torchinfo import summary

# summary(full_model)


# Convert test sequences to tensors
test_sequence_tensors = [digit_sequence_to_tensor(seq) for seq in test_seq_X]
test_sequence_tensors = torch.stack(test_sequence_tensors)  # Stack to form a batch tensor



# Set the model to evaluation mode
full_model.eval()

# Make predictions on the test set
predictions = []
with torch.no_grad():
    for test_sequence in test_sequence_tensors:
        test_sequence = test_sequence.unsqueeze(0)  # Add batch dimension
        output = full_model(test_sequence)
        prediction = torch.sigmoid(output).round().item()  # Get binary prediction (0/1)
        predictions.append(int(prediction))
        
        
        
# Save the predictions to a text file
with open('pred_textseq.txt', 'w') as f:
    for prediction in predictions:
        f.write(f'{prediction}\n')

print(f'Predictions saved to pred_textseq.txt')

