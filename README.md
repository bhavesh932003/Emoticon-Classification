# Emoticon-Classification

Objective:
Developed binary classification models to identify the best-performing model for each dataset, focusing on high accuracy and efficient use of training data.

Approach:

Leveraged LSTM and TCN models for feature extraction and classification, optimizing for sequence-based inputs across datasets.
Experimented with various fractions of training data (20% to 100%), achieving validation accuracies up to 98.57% for the best-performing model.
Optimized feature representations by removing redundancies and reducing parameters to meet the 10,000 trainable parameter limit.
Combined individual models using a weighted majority voting algorithm to improve generalization across datasets.
Outcome:
Achieved an overall validation accuracy of 93.66% with an efficient ensemble model within the parameter constraints.
