import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Multi hidden layer neural network with L2 regularization

# Step 1: Data Loading
df = pd.read_csv('train.csv')

# Step 2: Data Preprocessing
df['EJ'] = df['EJ'].map({'A': 0, 'B': 1})
df.fillna(df.median(numeric_only=True), inplace=True)

feature_columns = [col for col in df.columns if col not in ['Id', 'Class']]
df[feature_columns] = (df[feature_columns] - df[feature_columns].mean()) / df[feature_columns].std()

# Step 3: Data Splitting
X = df[feature_columns].values
y = df['Class'].values.reshape(-1, 1)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 4: Model Building
input_size = X_train.shape[1]
hidden_sizes = [50, 100, 50]  # Sets hidden layer sizes
output_size = 1
learning_rate = 0.01
lambda_l2 = 0.01

Ws = []
bs = []

prev_size = input_size
for size in hidden_sizes:
    Ws.append(np.random.randn(prev_size, size))
    bs.append(np.zeros((1, size)))
    prev_size = size

Ws.append(np.random.randn(prev_size, output_size))
bs.append(np.zeros((1, output_size)))

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

# Step 5: Training
epochs = 1000
losses = []
test_losses = []
basic_losses = []  # For visualization
basic_test_losses = []  # For visualization

for epoch in range(epochs):
    layer_outputs = [X_train]
    for i in range(len(hidden_sizes)):
        hidden_layer_input = np.dot(layer_outputs[-1], Ws[i]) + bs[i]
        hidden_layer_output = relu(hidden_layer_input)
        layer_outputs.append(hidden_layer_output)
    
    output_layer_input = np.dot(layer_outputs[-1], Ws[-1]) + bs[-1]
    predicted_output = sigmoid(output_layer_input)
    
    loss = np.mean((y_train - predicted_output)**2) + lambda_l2 * sum(np.sum(np.square(W)) for W in Ws)
    losses.append(loss)
    
    # Test set evaluation
    layer_outputs_test = [X_test]
    for i in range(len(hidden_sizes)):
        hidden_layer_input_test = np.dot(layer_outputs_test[-1], Ws[i]) + bs[i]
        hidden_layer_output_test = relu(hidden_layer_input_test)
        layer_outputs_test.append(hidden_layer_output_test)
    
    output_layer_input_test = np.dot(layer_outputs_test[-1], Ws[-1]) + bs[-1]
    predicted_output_test = sigmoid(output_layer_input_test)
    
    test_loss = np.mean((y_test - predicted_output_test)**2) + lambda_l2 * sum(np.sum(np.square(W)) for W in Ws)
    test_losses.append(test_loss)
    
    # Backpropagation
    output_error = y_train - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)
    
    deltas = [output_delta]
    for i in reversed(range(len(hidden_sizes))):
        hidden_layer_error = deltas[-1].dot(Ws[i+1].T)
        hidden_layer_delta = hidden_layer_error * relu_derivative(layer_outputs[i+1])
        deltas.append(hidden_layer_delta)
    
    deltas.reverse()
    
    for i in range(len(Ws)):
        Ws[i] += layer_outputs[i].T.dot(deltas[i]) * learning_rate - 2 * lambda_l2 * Ws[i]
        bs[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate

# Step 6: Evaluation
hidden_layer_input = np.dot(X_test, Ws[0]) + bs[0]
hidden_layer_output = relu(hidden_layer_input)

for i in range(1, len(hidden_sizes)):
    hidden_layer_input = np.dot(hidden_layer_output, Ws[i]) + bs[i]
    hidden_layer_output = relu(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, Ws[-1]) + bs[-1]
predicted_output = sigmoid(output_layer_input)
predicted_labels = (predicted_output > 0.5).astype(int)

accuracy = np.mean(predicted_labels == y_test)
print(f"Test Accuracy: {accuracy * 100}%")

# Step 7: Visualization
plt.plot(losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 8: Generate Submission Files

# Load test data
test_df = pd.read_csv('test.csv')

# Fix data
test_df['EJ'] = test_df['EJ'].map({'A': 0, 'B': 1})
test_df.fillna(test_df.median(numeric_only=True), inplace=True)

# Extract IDs
test_ids = test_df['Id'].values

# Preprocess test data
test_df[feature_columns] = (test_df[feature_columns] - df[feature_columns].mean()) / df[feature_columns].std()
X_test_submission = test_df[feature_columns].values  # Make sure to select only the feature columns

# Use the trained neural network to make predictions
layer_outputs_submission = [X_test_submission]
for i in range(len(hidden_sizes)):
    hidden_layer_input_submission = np.dot(layer_outputs_submission[-1], Ws[i]) + bs[i]
    hidden_layer_output_submission = relu(hidden_layer_input_submission)
    layer_outputs_submission.append(hidden_layer_output_submission)

output_layer_input_submission = np.dot(layer_outputs_submission[-1], Ws[-1]) + bs[-1]
predicted_output_submission = sigmoid(output_layer_input_submission)

# Format the predictions
submission_df = pd.DataFrame({
    'Id': test_ids,
    'class_0': 1 - predicted_output_submission.flatten(),
    'class_1': predicted_output_submission.flatten()
})

# Save to CSV files
submission_df.to_csv('svc_submission.csv', index=False)
submission_df.to_csv('dummy_submission.csv', index=False)