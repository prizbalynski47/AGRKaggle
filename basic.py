import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Data Loading
df = pd.read_csv('train.csv')

# Step 2: Data Preprocessing
# Handle categorical column "EJ"
df['EJ'] = df['EJ'].map({'A': 0, 'B': 1})

# remove rows with NaN values
df.dropna(inplace=True)

# Exclude 'Id' and 'Class' columns from feature columns for normalization
feature_columns = [col for col in df.columns if col not in ['Id', 'Class']]

# Normalization
df[feature_columns] = (df[feature_columns] - df[feature_columns].mean()) / df[feature_columns].std()


# Step 3: Data Splitting
# Splitting into features and labels
X = df[feature_columns].values
y = df['Class'].values.reshape(-1, 1)

# Splitting into training and testing sets (80% training, 20% testing)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 4: Model Building
# Initialize weights and biases
input_size = X_train.shape[1]
hidden_size = 100
output_size = 1
learning_rate = 0.01

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Step 5: Training
epochs = 1000
losses = []
test_losses = []

for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X_train, W1) + b1
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    predicted_output = sigmoid(output_layer_input)
    
    # Loss calculation
    loss = np.mean((y_train - predicted_output)**2)
    losses.append(loss)

    # Evaluate on test set
    hidden_layer_input_test = np.dot(X_test, W1) + b1
    hidden_layer_output_test = relu(hidden_layer_input_test)
    output_layer_input_test = np.dot(hidden_layer_output_test, W2) + b2
    predicted_output_test = sigmoid(output_layer_input_test)

    # Calculate test loss
    test_loss = np.mean((y_test - predicted_output_test)**2)
    test_losses.append(test_loss)
    
    # Backpropagation
    output_error = y_train - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)
    
    hidden_layer_error = output_delta.dot(W2.T)
    hidden_layer_delta = hidden_layer_error * relu_derivative(hidden_layer_output)
    
    # Update weights and biases
    W2 += hidden_layer_output.T.dot(output_delta) * learning_rate
    b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    W1 += X_train.T.dot(hidden_layer_delta) * learning_rate
    b1 += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

# Step 6: Evaluation
# Using the trained weights to make predictions on test data
hidden_layer_input = np.dot(X_test, W1) + b1
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, W2) + b2
predicted_output = sigmoid(output_layer_input)
predicted_labels = (predicted_output > 0.5).astype(int)

# Calculating accuracy
accuracy = np.mean(predicted_labels == y_test)
print(f"Test Accuracy: {accuracy * 100}%")

# Step 7: Visualization (comment out for kaggle submission)
plt.plot(losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Time')
plt.legend()
plt.show()

# Step 8: Generate Submission Files

# Load test data
test_df = pd.read_csv('test.csv')

# Handle categorical column "EJ"
test_df['EJ'] = test_df['EJ'].map({'A': 0, 'B': 1})

# Extract IDs
test_ids = test_df['Id'].values

# Preprocess test data
test_df[feature_columns] = (test_df[feature_columns] - df[feature_columns].mean()) / df[feature_columns].std()
X_test_submission = test_df[feature_columns].values  # Make sure to select only the feature columns

# Use the trained neural network to make predictions
hidden_layer_input = np.dot(X_test_submission, W1) + b1
hidden_layer_output = relu(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, W2) + b2
predicted_output = sigmoid(output_layer_input)

# Format the predictions
submission_df = pd.DataFrame({
    'Id': test_ids,
    'class_0': 1 - predicted_output.flatten(),
    'class_1': predicted_output.flatten()
})

# Save to CSV files
submission_df.to_csv('svc_submission.csv', index=False)
submission_df.to_csv('dummy_submission.csv', index=False)
