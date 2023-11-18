Basic NN no normalization:
NUM_HIDDEN_LAYERS = 1
HIDDEN_LAYER_SIZE = 50

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
DROPOUT_RATE = 0

NUM_EPOCHS = 100
Training Loss: 0.0000, Training Accuracy: 100.0000%, Validation Loss: 0.0035, Validation Accuracy: 90.3226%

Basic NN with normalization/new custom loss function:
NUM_HIDDEN_LAYERS = 1
HIDDEN_LAYER_SIZE = 50

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.5

NUM_EPOCHS = 200
adding norm: Training Loss: 0.0002, Training Accuracy: 98.1744%, Validation Loss: 0.0019, Validation Accuracy: 90.3226%
cust loss: Training Loss: 0.0002, Training Accuracy: 98.5801%, Validation Loss: 0.0024, Validation Accuracy: 91.9355%
