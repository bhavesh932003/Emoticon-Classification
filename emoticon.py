#importing necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input


# Load your datasets
train_data = pd.read_csv('datasets/train/train_emoticon.csv')  # Training dataset
val_data = pd.read_csv('datasets/valid/valid_emoticon.csv')  # Validation dataset
test_data = pd.read_csv('datasets/test/test_emoticon.csv')  # Test dataset


#preparing datasets
train_emoticons = train_data['input_emoticon'].values
train_labels = train_data['label'].values
val_emoticons = val_data['input_emoticon'].values
val_labels = val_data['label'].values



# Tokenize and pad sequences for data preprocessing
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(train_emoticons)

train_sequences = tokenizer.texts_to_sequences(train_emoticons)
val_sequences = tokenizer.texts_to_sequences(val_emoticons)

max_length = max(max(len(seq) for seq in train_sequences), max(len(seq) for seq in val_sequences))
train_padded_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post')
val_padded_sequences = pad_sequences(val_sequences, maxlen=max_length, padding='post')


#Feature Extraction using LSTM model
embedding_dim = 8
lstm_units = 32

# LSTM model architecture for binary classification
input_layer = Input(shape=(max_length,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim)(input_layer)
lstm_layer = LSTM(lstm_units, return_sequences=False)(embedding_layer)
dense_layer = Dense(16, activation='relu')(lstm_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)  # Change to 1 output with sigmoid activation
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



#Random forest classifier model
proportions = [0.2, 0.4, 0.6, 0.8, 1.0]
validation_accuracies = []
best_accuracy = 0
best_proportion = 0

for proportion in proportions:
    num_samples = int(len(train_data) * proportion)
    
    # Select training data
    X_train_emoticons = train_emoticons[:num_samples]
    y_train = train_labels[:num_samples]

    # Tokenize and pad the training data
    train_sequences = tokenizer.texts_to_sequences(X_train_emoticons)
    train_padded_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post')

    # Train the LSTM model on the current training subset
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_padded_sequences, y_train, epochs=20, batch_size=32, verbose=0)  # Set verbose=0 to suppress output

    # Extract features for the training subset
    train_features = model.predict(train_padded_sequences)
    val_features = model.predict(val_padded_sequences)

    # Convert features to DataFrames
    train_features_df = pd.DataFrame(train_features, columns=[f'feature_{i}' for i in range(train_features.shape[1])])
    val_features_df = pd.DataFrame(val_features, columns=[f'feature_{i}' for i in range(val_features.shape[1])])

    # Add features to the original DataFrames
    train_df_with_features = pd.concat([train_data[:num_samples].reset_index(drop=True), train_features_df], axis=1)
    val_df_with_features = pd.concat([val_data.reset_index(drop=True), val_features_df], axis=1)

    # Select features and labels for Random Forest
    X_train = train_df_with_features.iloc[:, -train_features_df.shape[1]:]  # Select last n columns for features
    y_train = train_df_with_features['label']

    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict labels on the validation set
    val_predictions = rf_classifier.predict(val_features_df)

    # Evaluate the model
    accuracy = accuracy_score(val_labels, val_predictions)
    validation_accuracies.append(accuracy)

    # Print results
    print(f'Validation Accuracy with {int(proportion * 100)}% training data: {accuracy:.2f}')
    print(classification_report(val_labels, val_predictions))
    
    # Checking for the best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_proportion = proportion


    # Confusion matrix
    confusion_mat = confusion_matrix(val_labels, val_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix with {int(proportion * 100)}% Training Data')
    plt.show()

    
    
# Ploting validation accuracy for different proportions
plt.figure(figsize=(10, 6))
plt.plot([f'{int(p * 100)}%' for p in proportions], validation_accuracies, marker='o')
plt.title('Validation Accuracy vs Training Data Proportion')
plt.xlabel('Training Data Proportion')
plt.ylabel('Validation Accuracy')
plt.ylim(0, 1)
plt.grid()
plt.show()



print(f'Best proportion: {best_proportion}, with validation accuracy: {best_accuracy:.2f}')


#training RF classifier on the best proprtion
num_samples_best = int(len(train_data) * best_proportion)
X_train_best = train_features_df.iloc[:num_samples_best]  # Select all columns for features
y_train_best = train_labels[:num_samples_best]

rf_classifier_best = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_best.fit(X_train_best, y_train_best)



test_data = pd.read_csv('datasets/test/test_emoticon.csv')  # Test dataset
test_emoticons = test_data['input_emoticon'].values

# Tokenize and pad the test sequences
test_sequences = tokenizer.texts_to_sequences(test_emoticons)
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Extract features from the test dataset using the trained LSTM model
test_features = model.predict(test_padded_sequences)

# Convert features to DataFrame
test_features_df = pd.DataFrame(test_features, columns=[f'feature_{i}' for i in range(test_features.shape[1])])

test_predictions = rf_classifier_best.predict(test_features_df)

# Adding predictions to the test DataFrame
test_data['predicted_label'] = test_predictions

# Saving predictions to a txt file
with open('pred_emoticon.txt', 'w') as f:
    for prediction in test_predictions:
        f.write(f'{prediction}\n')

print("Predictions saved to pred_emoticon.txt")
# Displaying the predictions
print(test_data[['input_emoticon', 'predicted_label']].head())



# model.summary()


