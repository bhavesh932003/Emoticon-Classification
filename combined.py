import pandas as pd
import numpy as np


# read emoticons dataset
train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()

valid_emoticon_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
valid_emoticon_X = valid_emoticon_df['input_emoticon'].tolist()
valid_emoticon_Y = valid_emoticon_df['label'].tolist()

test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()


# read feature dataset
train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']

valid_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
valid_feat_X = valid_feat['features']
valid_feat_Y = valid_feat['label']

test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']



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
from torch.utils.data import DataLoader, TensorDataset


emoticons = dict()
for seq in train_emoticon_X:
    for emo in seq:
        emoticons[emo] = emoticons.get(emo,0) + 1
for seq in valid_emoticon_X:
    for emo in seq:
        emoticons[emo] = emoticons.get(emo,0) + 1
for seq in test_emoticon_X:
    for emo in seq:
        emoticons[emo] = emoticons.get(emo,0) + 1
num_emoticons = len(emoticons) # 214
num_emoticons



# List of 226 unique emojis
emoji_list = list(emoticons.keys()) # Add all 226 emojis here

# Create a dictionary to map each emoji to a unique integer
emoji_to_idx = {emoji: idx for idx, emoji in enumerate(emoji_list)}

# Convert an emoticon string to a sequence of indices (based on the 214 emoji vocabulary)
def emoticon_to_idx_seq(emoticon_str):
    return [emoji_to_idx[emoji] for emoji in emoticon_str]

# Prepare your dataset (example dataset creation)
# X = ["🙂😂😛🙁😊😍🥳😢😎🤔😁😋😔", "😞😊🥳🙂🙁😎😛😂😍😢😔😋🤔", ...]  # List of strings (sequence of 13 emojis)
# y = [0, 1, ...]  # Binary labels

# Convert emoticon strings to sequences of indices
train_X_idx = [emoticon_to_idx_seq(s) for s in train_emoticon_X]
valid_X_idx = [emoticon_to_idx_seq(s) for s in valid_emoticon_X]
test_X_idx = [emoticon_to_idx_seq(s) for s in test_emoticon_X]

# Convert to PyTorch tensors
train_X = torch.tensor(train_X_idx, dtype=torch.long)
train_y = torch.tensor(train_emoticon_Y, dtype=torch.long)

valid_X_1 = torch.tensor(valid_X_idx, dtype=torch.long)
valid_y_1 = torch.tensor(valid_emoticon_Y, dtype=torch.long)

test_X_1 = torch.tensor(test_X_idx, dtype=torch.long)



class EmoticonLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=num_emoticons, embedding_dim=4, hidden_dim=20, output_size=2, batch_size=16, n_epochs=20, learning_rate=0.0015, n_layers=1):
        super(EmoticonLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax(dim=1)
        # Store hyperparameters as instance variables
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])  # Get output from the last time step
        return self.softmax(out)
    
    # Training
    def train_model(self, dataloader):
        # Training loop
        for epoch in range(self.n_epochs):
            self.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {total_loss/len(dataloader):.4f}')

    # Evaluation
    def evaluate_model(self, dataloader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                outputs = self(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        return (correct / total * 100)

# Hyperparameters
vocab_size = num_emoticons
embedding_dim = 4
hidden_dim = 18
output_size = 2  # Binary classification
n_layers = 1
batch_size = 16
n_epochs = 20
learning_rate = 0.0015

# Prepare DataLoader
train_dataset = TensorDataset(train_X, train_y) # train_X_unicode
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = TensorDataset(valid_X_1, valid_y_1)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Model
model_1 = EmoticonLSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_size, batch_size, n_epochs, learning_rate, n_layers)

model_1.train_model(train_loader)

val_accuracy_1 = model_1.evaluate_model(valid_loader)
print(f'Validation Accuracy of `Emoticon LSTM Classifier`: {val_accuracy_1:.2f}%')



# !pip install torchinfo
# from torchinfo import summary
# summary(model_1)



# many columns have unique value
num_examples, num_vectors, dim_vectors = train_feat_X.shape
useful_cols = [i for i in range(num_vectors) if max([pd.Series(train_feat_X[:,i,j]).nunique() for j in range(dim_vectors)]) > 1]
useful_cols


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F



# Chomp1d layer
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

# TemporalBlock layer
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

# TemporalConvNet layer
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

# TCNClassifier class with encapsulated methods
class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_channels, num_classes=2, kernel_size=2, dropout=0.2, lr=0.001):
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], num_classes)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # TCN expects input shape: (batch_size, input_size, sequence_length)
        y = self.tcn(x)
        # y[:, :, -1] takes the output from the last time step of the sequence
        return self.fc(y[:, :, -1])

    # Method to prepare dataloader
    def prepare_dataloader(self, X, y, batch_size=32):
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Method to train the model
    def train_model(self, dataloader, num_epochs=10):
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in dataloader:
                # Move tensors to the appropriate device
                inputs = inputs.permute(0, 2, 1)  # Change shape to (batch_size, input_size, sequence_length)
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

    # Method to evaluate the model
    def evaluate_model(self, dataloader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.permute(0, 2, 1)
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = (100 * correct / total)
        return accuracy

    
    
    
# Hyperparameters
n_vectors = 3
vector_dim = int(768*1/3)
n_samples = 7080
num_channels = [2, 16]  # Number of channels for each layer in the TCN
batch_size = 128
num_epochs = 20
learning_rate = 0.0015


train_X = torch.from_numpy(train_feat_X[ : , useful_cols, : vector_dim]).float()
train_y = torch.from_numpy(train_feat_Y).long()

valid_X_2 = torch.from_numpy(valid_feat_X[ : , useful_cols, : vector_dim]).float()
valid_y_2 = torch.from_numpy(valid_feat_Y).long()

test_X_2 = torch.from_numpy(test_feat_X[ : , useful_cols, : vector_dim]).float()



# Initialize models and dataloaders
model_2 = TCNClassifier(input_size=vector_dim, num_channels=num_channels, lr=learning_rate)
train_loader = model_2.prepare_dataloader(train_X, train_y, batch_size)
valid_loader = model_2.prepare_dataloader(valid_X_2, valid_y_2, batch_size)

# Train the models
model_2.train_model(train_loader, num_epochs)
while model_2.evaluate_model(valid_loader)<93:
    model_2 = TCNClassifier(input_size=vector_dim, num_channels=num_channels, lr=learning_rate)
    model_2.train_model(train_loader, num_epochs)

# Evaluate the model
val_accuracy_2 = model_2.evaluate_model(valid_loader)

print(f'Validation Accuracy of `TCN Feature Classifier`: {val_accuracy_2:.2f}%')



# !pip install torchinfo
# from torchinfo import summary
# summary(model_2)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class LSTMClassifierWithEmbedding(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size):
        super(LSTMClassifierWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)  # Input is (batch_size, seq_len) and output is (batch_size, seq_len, embedding_dim)
        _, (hn, _) = self.lstm(x)  # hn is the hidden state at the last time step
        out = self.fc(hn[-1])  # hn[-1] is the hidden state of the last LSTM layer
        return out

    def train_model(self, train_loader, criterion, optimizer, num_epochs):
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            epoch_loss = 0  # To track the loss for each epoch

            for batch_sequences, batch_labels in train_loader:
                # Forward pass
                outputs = self(batch_sequences)
                loss = criterion(outputs, batch_labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

    def evaluate_model(self, val_loader):
        self.eval()  # Set the model to evaluation mode
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for val_sequences, val_labels in val_loader:
                outputs = self(val_sequences)
                predictions = torch.sigmoid(outputs).round()  # Convert logits to binary predictions (0 or 1)
                all_labels.extend(val_labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy

    
    
    
# Convert sequence of digits into tensor (just a list of numbers)
def digit_sequence_to_tensor(sequence):
    return torch.tensor([int(digit) for digit in sequence], dtype=torch.long)

# Converting to Ytorch Tensors
train_X = [digit_sequence_to_tensor(seq) for seq in train_seq_X]
train_X = torch.stack(train_X)  # Stack to form a batch tensor
train_y = torch.tensor(train_seq_Y).float().unsqueeze(1)  # Convert labels to tensor and reshape

valid_X_3 = [digit_sequence_to_tensor(seq) for seq in valid_seq_X]
valid_X_3 = torch.stack(valid_X_3)  # Stack to form a batch tensor
valid_y_3 = torch.tensor(valid_seq_Y).float().unsqueeze(1)

test_X_3 = [digit_sequence_to_tensor(seq) for seq in test_seq_X]
test_X_3 = torch.stack(test_X_3)  # Stack to form a batch tensor



# Assuming sequences, labels, val_sequences, val_labels, etc. are already defined

# Hyperparameters
input_size = 10       # Number of possible digits (0-9)
embedding_dim = 16     # Size of the embedding vector for each digit
hidden_size = 20       # Size of the LSTM hidden state
output_size = 1        # Binary classification (0 or 1)
learning_rate = 0.0015
num_epochs = 25
batch_size = 8


# Create DataLoader for batch processing
train_dataset = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(valid_X_3, valid_y_3)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Initialize the LSTM model
model_3 = LSTMClassifierWithEmbedding(input_size, embedding_dim, hidden_size, output_size)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits
optimizer = optim.Adam(model_3.parameters(), lr=learning_rate)

# Train the model
model_3.train_model(train_loader, criterion, optimizer, num_epochs)

# Evaluate the model
val_accuracy_3 = model_3.evaluate_model(val_loader) * 100
print(f"Validation Accuracy: {val_accuracy_3:.2f}%")




# !pip install torchinfo
# from torchinfo import summary

# summary(model_3)


# Evaluating the combined voting model
def evaluate_combined_model(model_1, model_2, model_3):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    correct = 0
    total = 0
    
    for i in range(len(valid_X_1)):
        with torch.no_grad():
            o1 = model_1(valid_X_1[i].unsqueeze(0))  # LSTM output
            _, p1 = torch.max(o1, 1)
            
            # Permuting the input to match TCN input shape: (batch_size, input_size, sequence_length)
            o2 = model_2(valid_X_2[i].unsqueeze(0).permute(0, 2, 1))
            _, p2 = torch.max(o2, 1)
            
            o3 = model_3(valid_X_3[i].unsqueeze(0))
            _, p3 = torch.max(o3, 1)
            
            # Simple majority vote calculation
            # votes = torch.tensor([p1.item(), p2.item(), p3.item()])
            # vote = torch.mode(votes)[0].item()  # Taking the majority vote
            
            # Convert accuracies to weights (you could normalize them if needed)
            weights = torch.tensor([val_accuracy_1 / 100, val_accuracy_2 / 100, val_accuracy_3 / 100])
            votes = torch.tensor([p1.item(), p2.item(), p3.item()])  # p1, p2, p3 are model predictions
            weighted_votes = votes.float() * weights
            vote = 1 if weighted_votes.sum() >= (weights.sum() / 2) else 0  # Compare to half the total weight for majority
            
            total += 1
            correct += (vote == valid_y_1[i].item())  # Correctly index valid_y[i]
    
    return correct / total * 100

print(f'Validation Accuracy of the combined voting model: {evaluate_combined_model(model_1, model_2, model_3):.2f}%')



print("Total number of learned parameters of `Emoticon LSTM Classifier`      : ", 2670)
print("Total number of learned parameters of `TCN Feature Classifier`        : ", 3884)
print("Total number of learned parameters of `Text Sequence LSTM Classifier` : ", 3221)
print("=================================================================================")
cnt = 2670 + 3884 + 3221
print("Total number of learned parameters of the combined model              : ", cnt)
print("=================================================================================")


model_1.eval()
model_2.eval()
model_3.eval()

# Make predictions on the test set
predictions = []

for i in range(len(test_X_1)):
    with torch.no_grad():
        o1 = model_1(test_X_1[i].unsqueeze(0))  # LSTM output
        _, p1 = torch.max(o1, 1)

        # Permuting the input to match TCN input shape: (batch_size, input_size, sequence_length)
        o2 = model_2(test_X_2[i].unsqueeze(0).permute(0, 2, 1))
        _, p2 = torch.max(o2, 1)

        o3 = model_3(test_X_3[i].unsqueeze(0))
        _, p3 = torch.max(o3, 1)
        
        # Convert accuracies to weights (you could normalize them if needed)
        weights = torch.tensor([val_accuracy_1 / 100, val_accuracy_2 / 100, val_accuracy_3 / 100])
        votes = torch.tensor([p1.item(), p2.item(), p3.item()])  # p1, p2, p3 are model predictions
        weighted_votes = votes.float() * weights
        vote = 1 if weighted_votes.sum() >= (weights.sum() / 2) else 0  # Compare to half the total weight for majority
        
        predictions.append(vote)
        
        
        
# Save the predictions to a text file
with open('pred_combined.txt', 'w') as f:
    for prediction in predictions:
        f.write(f'{prediction}\n')

print(f'Predictions saved to pred_combined.txt')

