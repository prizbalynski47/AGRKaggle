import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# Step 1: Load Train and Test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Step 2: Preprocess Train Data
train_df['EJ'] = train_df['EJ'].map({'A': 0, 'B': 1})
train_df.fillna(train_df.median(numeric_only=True), inplace=True)

feature_columns = [col for col in train_df.columns if col not in ['Id', 'Class']]
train_features = train_df[feature_columns]

# Normalize the features
mean = train_features.mean()
std = train_features.std()
train_features = (train_features - mean) / std

# Extract features and labels
X = train_features.values
y = train_df['Class'].values.reshape(-1, 1)

# Step 3: Preprocess Test Data
# Apply the same mapping and fillna
test_df['EJ'] = test_df['EJ'].map({'A': 0, 'B': 1})
test_df.fillna(train_df.median(numeric_only=True), inplace=True)  # Use median from train_df

# Normalize using mean and std from train data
test_features = test_df[feature_columns]
test_features = (test_features - mean) / std

# Extract features for test data
X_test = test_features.values

# Step 3: Data Splitting
X_train = X
y_train = y
# 4.1 Correlation Matrix
corr_matrix = train_features.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# 4.2 Feature Importance using Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train.ravel())  # Flatten y_train for sklearn compatibility
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [feature_columns[i] for i in indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# 4.3 Mutual Information
mi = mutual_info_classif(X_train, y_train.ravel())
mi /= np.max(mi)  # Normalize

plt.figure(figsize=(12, 6))
plt.title("Normalized Mutual Information")
plt.bar(range(len(mi)), mi[indices], align="center")
plt.xticks(range(len(mi)), [feature_columns[i] for i in indices], rotation=90)
plt.xlim([-1, len(mi)])
plt.show()