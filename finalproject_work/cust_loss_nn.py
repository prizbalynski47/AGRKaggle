import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

# Simple adjustable nn

WORST_FEATURES_TO_REMOVE = 0
NUM_OF_CATS_TO_CHANGE = 50
NOISE_SCALE = 1
NUM_COPIES = 19

NUM_HIDDEN_LAYERS = 3
HIDDEN_LAYER_SIZE = [32, 16, 8]

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.007
DROPOUT_RATE = [0.75, 0.5, 0.25]

NUM_EPOCHS = 1000
K_FOLDS = 5
TRAIN_PER_FOLD = 10
SAVED_ATTEMPTS_PER_FOLD = 2
PATIENCE = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Step 1: Load Train and Test data
# train_df = pd.read_csv('train.csv')

# # Shuffle the DataFrame
# train_df = train_df.sample(frac=1, random_state=42)  # Setting random_state for reproducibility
# train_df.reset_index(drop=True, inplace=True)

# # Step 2: Preprocess Train Data
# train_df['EJ'] = train_df['EJ'].map({'A': 0, 'B': 1})
# train_df.fillna(train_df.median(numeric_only=True), inplace=True)

# # Identify categorical and numerical columns
# categorical_columns = ['EJ', 'Class']  # List of all categorical columns
# numerical_columns = [col for col in train_df.columns if col not in categorical_columns + ['Id']]
# categorical_columns = ['EJ']
# # Normalize the original data
# mean = train_df[numerical_columns].mean()
# std = train_df[numerical_columns].std()
# train_df[numerical_columns] = (train_df[numerical_columns] - mean) / std

# # Duplicate the DataFrame
# noisy_df = train_df.copy()

# def flip_categorical(value):
#     return 1 - value

# # List to store the rows
# combined_rows = []

# # Modification for Gaussian noise
# for index, row in train_df.iterrows():
#     # Append the original row to the list
#     combined_rows.append(row)
#     for i in range(NUM_COPIES):
#         # Create a noisy version of the row
#         noisy_row = row.copy()
#         selected_columns = np.random.choice(categorical_columns + numerical_columns, size=NUM_OF_CATS_TO_CHANGE, replace=False)
#         for col in selected_columns:
#             if col in categorical_columns:
#                 noisy_row[col] = flip_categorical(row[col])
#             else:
#                 class_category = row['Class']
#                 class_std_dev = train_df[train_df['Class'] == class_category][col].std()
#                 # Add Gaussian noise
#                 gaussian_noise = np.random.normal(0, class_std_dev * NOISE_SCALE)
#                 noisy_row[col] += gaussian_noise

#         # Append the noisy row to the list
#         combined_rows.append(noisy_row)

# # Convert the list of rows to a DataFrame
# combined_df = pd.concat(combined_rows, axis=1).transpose()

# # Reset index of the combined DataFrame
# combined_df.reset_index(drop=True, inplace=True)

combined_df = pd.read_csv('modified_combined_data.csv')

# Extract features and labels
feature_columns = [col for col in combined_df.columns if col not in ['Id', 'Class']]
X = combined_df[feature_columns].values
y = combined_df['Class'].values.reshape(-1, 1)

# Calculate the indices of the non-noisy data
non_noisy_indices = []
for i in range(len(combined_df) // (NUM_COPIES + 1)):
    # Every NUM_COPIES + 1 index is the start of a new original data point
    non_noisy_indices.extend(range(i * (NUM_COPIES + 1), (i + 1) * (NUM_COPIES + 1)))

y_values = combined_df['Class'].values
y_values = pd.to_numeric(y_values, errors='coerce')

# Train a Random Forest Classifier to get feature importances
rf = RandomForestClassifier()
rf.fit(X, y_values)

# Get feature importances and sort them
importances = rf.feature_importances_
indices = np.argsort(importances)

# Remove the least important features
features_to_remove = [feature_columns[i] for i in indices[:WORST_FEATURES_TO_REMOVE]]
feature_columns = [col for col in feature_columns if col not in features_to_remove]

# Update train_features to exclude the least important features
train_features = combined_df[feature_columns]
X = train_features.values

# Step 3: Preprocess Test Data
test_df = pd.read_csv('test.csv')

test_df['EJ'] = test_df['EJ'].map({'A': 0, 'B': 1})
# test_df.fillna(train_df.median(numeric_only=True), inplace=True)

# # Normalize using mean and std from train data
# test_df[numerical_columns] = (test_df[numerical_columns] - mean) / std

test_features = test_df[feature_columns]

# Extract features for test data
X_test = test_features.values
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

kf = KFold(n_splits=K_FOLDS, shuffle=False) # True , random_state=42)

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

    # Check for zero division
    if n_class_0 == 0: n_class_0 = 1
    if n_class_1 == 0: n_class_1 = 1
        
    # Averaging the loss for each class
    loss = (torch.sum(loss_class_0) / n_class_0 + torch.sum(loss_class_1) / n_class_1) / 2

    return loss

# Model Structure
class BasicNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=NUM_HIDDEN_LAYERS, dropout_rate=DROPOUT_RATE):
        super(BasicNN, self).__init__()

        layers = [nn.Linear(input_size, hidden_size[0]), nn.ReLU(), nn.Dropout(dropout_rate[0])]

        for i in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size[i], hidden_size[i+1]), nn.ReLU(), nn.Dropout(dropout_rate[i+1])]

        layers += [nn.Linear(hidden_size[-1], output_size), nn.Sigmoid()]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

models = []
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
val_losses_full = []
best_attempt_data = {"val_loss": 0, "val_accuracies": [0]}

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    # Initialize variables to track the best attempt
    best_val_loss = float('inf')
    best_attempts_heap = []

    for attempt in range(TRAIN_PER_FOLD):
        sys.stdout.write(f"\rFold {fold + 1}/{K_FOLDS}, Attempt {attempt + 1}/{TRAIN_PER_FOLD}. Last Loss: {best_attempt_data['val_loss']}, Last Accuracy: {best_attempt_data['val_accuracies'][-1]}")
        sys.stdout.flush()
        
        attempt_best_val_loss = float('inf')
        epochs_without_improvement = 0

        temp_train_losses = []
        temp_val_losses = []
        temp_train_accuracies = []
        temp_val_accuracies = []

        # Filter out only the non-noisy indices for validation
        val_non_noisy_indices = [index for index in val_index if index in non_noisy_indices]

        # Extract training and non-noisy validation data
        X_train_fold, X_val_fold = X[train_index], X[val_non_noisy_indices]
        y_train_fold, y_val_fold = y[train_index], y[val_non_noisy_indices]

        if isinstance(X_train_fold, pd.DataFrame):
            X_train_fold = X_train_fold.values.astype(np.float32)
        elif isinstance(X_train_fold, np.ndarray):
            X_train_fold = X_train_fold.astype(np.float32)
        if isinstance(X_val_fold, pd.DataFrame):
            X_val_fold = X_val_fold.values.astype(np.float32)
        elif isinstance(X_val_fold, np.ndarray):
            X_val_fold = X_val_fold.astype(np.float32)
        if isinstance(y_train_fold, pd.DataFrame):
            y_train_fold = y_train_fold.values.astype(np.float32)
        elif isinstance(y_train_fold, np.ndarray):
            y_train_fold = y_train_fold.astype(np.float32)
        if isinstance(y_val_fold, pd.DataFrame):
            y_val_fold = y_val_fold.values.astype(np.float32)
        elif isinstance(y_val_fold, np.ndarray):
            y_val_fold = y_val_fold.astype(np.float32)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32)

        # DataLoader for the current fold
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

        # Initialize model for the current fold
        model = BasicNN(X_train_fold.shape[1], HIDDEN_LAYER_SIZE, 1)
        model.to(device)
        criterion = balanced_log_loss
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.1)
        # Train the model
        for epoch in range(NUM_EPOCHS):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validation phase
            model.eval()
            total_train_loss = 0
            with torch.no_grad():
                X_train_tensor = X_train_tensor.to(device)
                y_train_tensor = y_train_tensor.to(device)
                y_train_pred = model(X_train_tensor)
                y_train_pred_class = y_train_pred.round().flatten()
                correct_train = (y_train_pred_class == y_train_tensor.flatten()).sum().item()
                train_accuracy = correct_train / len(y_train_tensor) * 100

                # Training loss
                train_loss = criterion(y_train_pred, y_train_tensor)
                total_train_loss += train_loss.item()
            avg_train_loss = total_train_loss / len(y_train_tensor)

            total_val_loss = 0
            with torch.no_grad():
                X_val_tensor = X_val_tensor.to(device)
                y_val_tensor = y_val_tensor.to(device)
                y_val_pred = model(X_val_tensor)
                y_val_pred_class = y_val_pred.round().flatten()
                correct_val = (y_val_pred_class == y_val_tensor.flatten()).sum().item()
                val_accuracy = correct_val / len(y_val_tensor) * 100

                # Validation loss
                val_loss = criterion(y_val_pred, y_val_tensor)
                total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(y_val_tensor)

            temp_train_losses.append(avg_train_loss)
            temp_val_losses.append(avg_val_loss)
            temp_train_accuracies.append(train_accuracy)
            temp_val_accuracies.append(val_accuracy)

            # Check if the current validation loss is lower than the best known
            if avg_val_loss < attempt_best_val_loss:
                attempt_best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_attempt_data = {
                "val_loss": avg_val_loss,
                "model_state": model.state_dict().copy(),
                "train_losses": temp_train_losses.copy(),
                "val_losses": temp_val_losses.copy(),
                "train_accuracies": temp_train_accuracies.copy(),
                "val_accuracies": temp_val_accuracies.copy()
                }
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE: break
        
        val_losses_full.append(best_attempt_data["val_loss"])
        # Check if the current attempt is one of the best
        if len(best_attempts_heap) < SAVED_ATTEMPTS_PER_FOLD:
            best_attempts_heap.append(best_attempt_data)
        else:
            worst_best_attempt = max(best_attempts_heap, key=lambda x: x["val_loss"])
            if avg_val_loss < worst_best_attempt["val_loss"]:
                best_attempts_heap.remove(worst_best_attempt)
                best_attempts_heap.append(best_attempt_data)

    # Save the best attempts data for the current fold
    for attempt_data in best_attempts_heap:
        models.append(attempt_data["model_state"])
        train_losses.append(attempt_data["train_losses"])
        val_losses.append(attempt_data["val_losses"])
        train_accuracies.append(attempt_data["train_accuracies"])
        val_accuracies.append(attempt_data["val_accuracies"])
print()

# Find the maximum number of epochs trained across all folds
max_epochs = max(len(losses) for losses in train_losses)

# Extend shorter lists by repeating their last value
for f in range(K_FOLDS * SAVED_ATTEMPTS_PER_FOLD):
    last_train_loss = train_losses[f][-1]
    last_val_loss = val_losses[f][-1]
    last_train_accuracy = train_accuracies[f][-1]
    last_val_accuracy = val_accuracies[f][-1]

    train_losses[f].extend([last_train_loss] * (max_epochs - len(train_losses[f])))
    val_losses[f].extend([last_val_loss] * (max_epochs - len(val_losses[f])))
    train_accuracies[f].extend([last_train_accuracy] * (max_epochs - len(train_accuracies[f])))
    val_accuracies[f].extend([last_val_accuracy] * (max_epochs - len(val_accuracies[f])))

# Calculate averages for each epoch
avg_train_losses = [np.mean([train_losses[f][epoch] for f in range(K_FOLDS)]) for epoch in range(max_epochs)]
avg_val_losses = [np.mean([val_losses[f][epoch] for f in range(K_FOLDS)]) for epoch in range(max_epochs)]
avg_train_accuracies = [np.mean([train_accuracies[f][epoch] for f in range(K_FOLDS)]) for epoch in range(max_epochs)]
avg_val_accuracies = [np.mean([val_accuracies[f][epoch] for f in range(K_FOLDS)]) for epoch in range(max_epochs)]

print(f'Averages, Training Loss: {avg_train_losses[-1]:.7f}, Training Accuracy: {avg_train_accuracies[-1]:.4f}%, Validation Loss: {avg_val_losses[-1]:.7f}, Validation Accuracy: {avg_val_accuracies[-1]:.4f}%')
print(f"Full average val loss: {np.mean(val_losses_full)}")

# Aggregate predictions from each fold
test_predictions = []

for state_dict in models:
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        fold_prediction = model(X_test_tensor)
        test_predictions.append(fold_prediction)

# Average the predictions
avg_test_predictions = torch.mean(torch.stack(test_predictions), dim=0)

# Convert predictions to numpy array
avg_test_predictions = avg_test_predictions.cpu()
final_predictions = avg_test_predictions.numpy()

# Calculate the total estimated probability for each class
class_0_est_instances = np.sum(1 - final_predictions[:, 0])
class_1_est_instances = np.sum(final_predictions[:, 0])

# Adjust the probabilities
adjusted_class_0_probs = (1 - final_predictions[:, 0]) / class_0_est_instances
adjusted_class_1_probs = final_predictions[:, 0] / class_1_est_instances

# Stack the adjusted probabilities
adjusted_probs = np.stack([adjusted_class_0_probs, adjusted_class_1_probs], axis=1)

# Normalize the probabilities so that they sum up to 1 for each instance
normalized_probs = adjusted_probs / np.sum(adjusted_probs, axis=1, keepdims=True)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'Id': test_df['Id'],
    'class_0': normalized_probs[:, 0],
    'class_1': normalized_probs[:, 1]
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

# Plotting
plt.figure(figsize=(12, 5))

# Plot for average loss
plt.subplot(1, 2, 1)
plt.plot(avg_train_losses, label='Average Training Loss')
plt.plot(avg_val_losses, label='Average Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Training and Validation Loss Over Epochs')
plt.legend()

# Plot for average accuracy
plt.subplot(1, 2, 2)
plt.plot(avg_train_accuracies, label='Average Training Accuracy')
plt.plot(avg_val_accuracies, label='Average Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Average Training and Validation Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()