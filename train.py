import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from lbp_feature_extraction import extract_lbp_features
from cnn_model import build_model
from eho_optimization import eho_feature_selection

# Define paths
data_dir = "data/processed/"
categories = ["healthy", "diseased"]

# Load dataset
print("Loading dataset...")
images, labels = [], []
for label, category in enumerate(categories):
    category_path = os.path.join(data_dir, category)
    print(f"Processing category: {category}")
    for file in os.listdir(category_path):
        file_path = os.path.join(category_path, file)
        if not os.path.isfile(file_path):  # Check if the file exists
            print(f"File not found: {file_path}")
            continue
        print(f"Processing file: {file_path}")
        try:
            lbp_features = extract_lbp_features(file_path)
            images.append(lbp_features)
            labels.append(label)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

# Convert to arrays
print("Converting data to arrays...")
images = np.array(images)
labels = np.array(labels)

# Split dataset
print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Do NOT use to_categorical for binary classification with sigmoid
# y_train = to_categorical(y_train)  # Comment out or remove this line
# y_test = to_categorical(y_test)    # Comment out or remove this line

# Build model
print("Building model...")
model = build_model(input_shape=(X_train.shape[1],))  # Input shape is the number of LBP features

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Use EHO for feature selection (example template)
print("Performing EHO feature selection...")
def fitness_function(features):
    # Define a simple fitness function (e.g., accuracy from a classifier)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(features, y_train)  # Use training data for feature selection
    return clf.score(features, y_train)  # Evaluate on training data

# Perform EHO feature selection
selected_features_mask = eho_feature_selection(X_train, fitness_function)
print("Selected features mask:", selected_features_mask)

# Train the model with selected features
print("Training the model with selected features...")
X_train_selected = X_train[:, selected_features_mask]
X_test_selected = X_test[:, selected_features_mask]

# Build and train the model with selected features
model_selected = build_model(input_shape=(X_train_selected.shape[1],))
history_selected = model_selected.fit(X_train_selected, y_train, epochs=10, validation_data=(X_test_selected, y_test))

# Evaluate the model with selected features
print("Evaluating the model with selected features...")
loss_selected, accuracy_selected = model_selected.evaluate(X_test_selected, y_test)
print(f"Test Loss (Selected Features): {loss_selected}")
print(f"Test Accuracy (Selected Features): {accuracy_selected}")