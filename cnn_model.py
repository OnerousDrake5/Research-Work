import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape):
    """
    Build a fully connected neural network for feature classification.
    :param input_shape: Shape of the input features (e.g., (10,)).
    :return: Compiled model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),  # Input layer
        layers.Dense(128, activation='relu'),  # Fully connected layer
        layers.Dropout(0.5),  # Dropout for regularization
        layers.Dense(64, activation='relu'),  # Another fully connected layer
        layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model