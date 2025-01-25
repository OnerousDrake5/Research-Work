import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess an image for the CNN.
    :param image_path: Path to the image file.
    :param target_size: Target size of the image (height, width).
    :return: Preprocessed image as a numpy array.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}. Check if the file is a valid image.")
    
    # Resize the image
    image = cv2.resize(image, target_size)
    
    # Normalize the pixel values to the range [0, 1]
    image = image / 255.0
    
    # Add a channel dimension (required for CNN input)
    image = np.expand_dims(image, axis=-1)
    
    return image

# Function to load the dataset
def load_dataset(data_dir, categories, target_size=(128, 128)):
    """
    Load and preprocess the dataset for the CNN.
    :param data_dir: Path to the dataset directory.
    :param categories: List of categories (e.g., ["healthy", "diseased"]).
    :param target_size: Target size of the images (height, width).
    :return: Tuple of (images, labels) as numpy arrays.
    """
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
                image = load_and_preprocess_image(file_path, target_size)
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Function to build a CNN model
def build_cnn(input_shape):
    """
    Build a CNN model for binary classification.
    :param input_shape: Shape of the input images (height, width, channels).
    :return: Compiled CNN model.
    """
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten the feature maps
        layers.Flatten(),
        
        # Fully connected layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout for regularization
        layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Main function
def main():
    # Define paths
    data_dir = "data/processed/"
    categories = ["healthy", "diseased"]
    
    # Load and preprocess the dataset
    print("Loading dataset...")
    images, labels = load_dataset(data_dir, categories)
    
    # Split dataset into training and testing sets
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Build the CNN model
    print("Building CNN model...")
    input_shape = X_train.shape[1:]  # Shape of the input images (height, width, channels)
    cnn_model = build_cnn(input_shape)
    
    # Train the CNN model
    print("Training the CNN model...")
    history = cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    # Evaluate the CNN model
    print("Evaluating the CNN model...")
    loss, accuracy = cnn_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    
    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()