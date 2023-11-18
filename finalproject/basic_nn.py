import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create data loaders
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Model Structure
class BasicNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1, dropout_rate=0):
        super(BasicNN, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]

        for i in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]

        layers += [nn.Linear(hidden_size, output_size), nn.Sigmoid()]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

model = BasicNN(X_train.shape[1], 50, 1)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):  # number of epochs
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_val_pred = model(X_val_tensor)
    y_val_pred_class = y_val_pred.round()
    y_val_pred_class = y_val_pred_class.flatten().type_as(y_val_tensor)
    correct = (y_val_pred_class == y_val_tensor.flatten()).sum().item()
    accuracy = correct / len(y_val_tensor) * 100
    print(f'Validation Accuracy: {accuracy}%')

# Make predictions
with torch.no_grad():
    test_probabilities = model(X_test_tensor).numpy()

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'Id': test_df['Id'],
    'class_0': 1 - test_probabilities[:, 0],
    'class_1': test_probabilities[:, 0]
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)