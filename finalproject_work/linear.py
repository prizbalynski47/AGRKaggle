import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Simple linear model

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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train.ravel())

# Make predictions on the validation set and calculate accuracy
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy * 100}%')

# Make predictions on the test set
test_probabilities = model.predict_proba(X_test)

# Create the submission DataFrame using the test data IDs
submission_df = pd.DataFrame({
    'Id': test_df['Id'],
    'class_0': test_probabilities[:, 0],
    'class_1': test_probabilities[:, 1]
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)