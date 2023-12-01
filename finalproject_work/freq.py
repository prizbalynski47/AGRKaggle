import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Frequency dummy submission

# Step 1: Data Loading
df = pd.read_csv('train.csv')

# Step 2: Data Preprocessing
df['EJ'] = df['EJ'].map({'A': 0, 'B': 1})
df.fillna(df.median(numeric_only=True), inplace=True)

feature_columns = [col for col in df.columns if col not in ['Id', 'Class']]
df[feature_columns] = (df[feature_columns] - df[feature_columns].mean()) / df[feature_columns].std()

X = df[feature_columns].values
y = df['Class'].values.reshape(-1, 1)

# Step 3: Data Splitting
X_train = X
y_train = y

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate Frequencies
class_counts = y_train.flatten().tolist().count
class_0_freq = class_counts(0) / len(y_train)
class_1_freq = class_counts(1) / len(y_train)

# Load the test data
test_df = pd.read_csv('test.csv')

# Create the submission DataFrame using the test data IDs
submission_df = pd.DataFrame({
    'Id': test_df['Id'],
    'class_0': [class_0_freq] * len(test_df),
    'class_1': [class_1_freq] * len(test_df)
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

# Validation accuracy
test_class_counts = y_val.flatten().tolist().count
test_class_0_freq = test_class_counts(0) / len(y_val)
test_class_1_freq = test_class_counts(1) / len(y_val)

if class_0_freq > class_1_freq:
    accuracy = test_class_0_freq
else:
    accuracy = test_class_1_freq

print(f'Accuracy: {accuracy}%')