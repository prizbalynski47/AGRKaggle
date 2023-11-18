import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Simple adjustable nn

NUM_HIDDEN_LAYERS = 40
HIDDEN_LAYER_SIZE = 20

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
DROPOUT_RATE = 0.1
DROPOUT_INIT = 0.6

NUM_EPOCHS = 200

NUM_OUTLIERS = 50


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

def add_outliers_with_labels(features, labels, num_outliers, magnitude=10, outlier_label=1):
    """
    Adds outliers to the dataset and corresponding labels.
    features: numpy array of features.
    labels: numpy array of labels.
    num_outliers: number of outliers to add.
    magnitude: the magnitude of the outliers.
    outlier_label: the label for the outliers.
    """
    # Randomly select indices to introduce outliers
    outlier_indices = np.random.choice(features.shape[0], num_outliers, replace=False)

    # Randomly select features to modify
    feature_indices = np.random.choice(features.shape[1], num_outliers, replace=True)

    # Add or subtract the magnitude to create outliers
    for i, feature_index in zip(outlier_indices, feature_indices):
        if np.random.rand() > 0.5:
            features[i, feature_index] += magnitude
        else:
            features[i, feature_index] -= magnitude

        # Set the corresponding label to the outlier label
        labels[i] = outlier_label

    return features, labels

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Selection based on Importance and Mutual Information

# 4.1 Calculate Feature Importance using Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train.ravel())  # Flatten y_train for sklearn compatibility
feature_importances = rf.feature_importances_

# Normalize both metrics for comparison
feature_importances_normalized = feature_importances / feature_importances.max()

# Combine metrics (consider different strategies)
combined_metric = feature_importances_normalized

# Set a threshold for feature selection based on your analysis
threshold = np.median(combined_metric) * 0.6 # Adjust this based on your analysis

# Select features above the threshold
selected_features = [feature_columns[i] for i in range(len(combined_metric)) if combined_metric[i] > threshold]

# Apply feature selection to all datasets
X_train = X_train[:, [feature_columns.index(f) for f in selected_features]]
X_val = X_val[:, [feature_columns.index(f) for f in selected_features]]
X_test = X_test[:, [feature_columns.index(f) for f in selected_features]]

X_train, y_train = add_outliers_with_labels(X_train.copy(), y_train.copy(), NUM_OUTLIERS)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create data loaders
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Custom Loss
def balanced_log_loss(output, target, epsilon=1e-7):
    # Clipping the predicted probabilities
    output = torch.clamp(output, epsilon, 1 - epsilon)

    # Calculating log loss for each class
    loss_class_0 = - (1 - target) * torch.log(1 - output)
    loss_class_1 = - target * torch.log(output)

    # Counting the number of observations in each class
    n_class_0 = torch.sum(1 - target)
    n_class_1 = torch.sum(target)

    # Averaging the loss for each class
    loss = (torch.sum(loss_class_0) / n_class_0 + torch.sum(loss_class_1) / n_class_1) / 2

    return loss

# Model Structure
class BasicNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=NUM_HIDDEN_LAYERS, dropout_rate=DROPOUT_RATE):
        super(BasicNN, self).__init__()

        self.layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        # First layer
        self.layers.append(nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(DROPOUT_INIT)))
        # Skip connection for the first layer (if needed)
        self.initial_skip = nn.Linear(input_size, hidden_size)

        # Intermediate layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)))
            self.skip_connections.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Sigmoid())

    def forward(self, x):
        skip_input = self.initial_skip(x)  # Transform input to match hidden layer size

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i > 0:  # Skip connections start from the second layer
                skip = self.skip_connections[i-1](skip_input)
                x = x + skip  # Element-wise addition
            skip_input = x  # Update skip_input for the next iteration

        x = self.output_layer(x)
        return x

model = BasicNN(X_train.shape[1], HIDDEN_LAYER_SIZE, 1)

# Loss and optimizer
criterion = balanced_log_loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.9, 10)

# Lists to store metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_loss = float('inf')
best_model_state = None

# Train the model
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation phase
    model.eval()
    total_train_loss = 0
    with torch.no_grad():
        y_train_pred = model(X_train_tensor)
        y_train_pred_class = y_train_pred.round().flatten()
        correct_train = (y_train_pred_class == y_train_tensor.flatten()).sum().item()
        train_accuracy = correct_train / len(y_train_tensor) * 100
        train_accuracies.append(train_accuracy)

        # Training loss
        train_loss = criterion(y_train_pred, y_train_tensor)
        total_train_loss += train_loss.item()
    avg_train_loss = total_train_loss / len(y_train_tensor)
    train_losses.append(avg_train_loss)

    total_val_loss = 0
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        y_val_pred_class = y_val_pred.round().flatten()
        correct_val = (y_val_pred_class == y_val_tensor.flatten()).sum().item()
        val_accuracy = correct_val / len(y_val_tensor) * 100
        val_accuracies.append(val_accuracy)

        # Validation loss
        val_loss = criterion(y_val_pred, y_val_tensor)
        total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(y_val_tensor)
    val_losses.append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
    
    scheduler.step(avg_val_loss)

print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}%, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}%')
print(f'Best val loss: {best_val_loss:.4f}')
model.load_state_dict(best_model_state)

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

# Plotting
plt.figure(figsize=(12, 5))

# Plot for loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()

# Plot for accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()