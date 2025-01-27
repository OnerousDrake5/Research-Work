import os
import numpy as np
import cv2
import time
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import set_global_policy
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Enable mixed precision for faster training on GPU
set_global_policy('mixed_float16')

# Function to compute LBP features
def compute_lbp(image, P=8, R=1):
    image = (image * 255).astype(np.uint8)  # Convert to integer dtype
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}. Check if the file is a valid image.")
    image = cv2.resize(image, target_size)
    image = image / 255.0
    lbp_features = compute_lbp(image)  # Compute LBP features
    image = np.expand_dims(image, axis=-1)
    return image, lbp_features

# Function to load dataset
def load_dataset(data_dir, categories, target_size=(128, 128)):
    images, lbp_features, labels = [], [], []
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        print(f"Loading images from: {category_path}")  # Debug statement
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            if not os.path.isfile(file_path):
                continue
            try:
                image, lbp = load_and_preprocess_image(file_path, target_size)
                images.append(image)
                lbp_features.append(lbp)
                labels.append(label)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    print(f"Loaded {len(images)} images.")  # Debug statement
    return np.array(images), np.array(lbp_features), np.array(labels)

# CNN model
def build_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),  # Add Input layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten()
    ])
    return model

# EHO feature selection
def eho_feature_selection(features, y_train, fitness_function, num_iterations=5, population_size=5, timeout=300):
    start_time = time.time()
    num_features = features.shape[1]
    population = np.random.randint(2, size=(population_size, num_features))
    fitness_history = []  # Store fitness values over generations

    for iteration in range(num_iterations):
        if time.time() - start_time > timeout:
            print("Timeout reached. Stopping EHO.")
            break
        print(f"Iteration {iteration + 1}/{num_iterations}")
        fitness_scores = []
        for i, individual in enumerate(population):
            selected_features = features[:, individual == 1]
            score = fitness_function(selected_features, y_train)
            fitness_scores.append(score)
        best_indices = np.argsort(fitness_scores)[-population_size//2:]
        population = population[best_indices]
        new_population = []
        for i in range(population_size):
            parent1, parent2 = np.random.choice(len(best_indices), 2, replace=False)
            child = population[parent1] | population[parent2]
            if np.random.rand() < 0.1:
                child[np.random.randint(num_features)] = 1 - child[np.random.randint(num_features)]
            new_population.append(child)
        population = np.array(new_population)
        fitness_history.append(np.max(fitness_scores))  # Store the best fitness value

    best_individual = population[np.argmax(fitness_scores)]
    return best_individual == 1, fitness_history

# Simplified fitness function using mutual information
def fitness_function(features, y_train):
    scores = mutual_info_classif(features, y_train)
    return np.mean(scores)

# Final model with regularization
def build_final_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function
def main():
    data_dir = "data/processed/"
    categories = ["healthy", "diseased"]
    images, lbp_features, labels = load_dataset(data_dir, categories)
    X_train, X_test, X_train_lbp, X_test_lbp, y_train, y_test = train_test_split(
        images, lbp_features, labels, test_size=0.2, random_state=42
    )

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # CNN for feature extraction
    cnn_model = build_cnn((128, 128, 1))
    print("Extracting features from training set...")  # Debug statement
    X_train_features = cnn_model.predict(X_train, batch_size=32, verbose=1)  # Verbose output
    print("Extracting features from test set...")  # Debug statement
    X_test_features = cnn_model.predict(X_test, batch_size=32, verbose=1)  # Verbose output

    # Concatenate CNN and LBP features
    X_train_combined = np.hstack((X_train_features, X_train_lbp))
    X_test_combined = np.hstack((X_test_features, X_test_lbp))

    # Use PCA for dimensionality reduction
    pca = PCA(n_components=50)  # Reduce to 50 components
    X_train_pca = pca.fit_transform(X_train_combined)
    X_test_pca = pca.transform(X_test_combined)

    # EHO for feature selection
    selected_features_mask, fitness_history = eho_feature_selection(X_train_pca, y_train, fitness_function)
    X_train_selected = X_train_pca[:, selected_features_mask]
    X_test_selected = X_test_pca[:, selected_features_mask]

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    # Final model training
    final_model = build_final_model((X_train_selected.shape[1],))
    history = final_model.fit(
        X_train_resampled, y_train_resampled,
        epochs=20,
        batch_size=32,
        validation_data=(X_test_selected, y_test),
        class_weight=class_weights,
        verbose=1
    )

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Convergence Curve: Loss vs Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Convergence Curve: Accuracy vs Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot EHO fitness history
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history, label='Best Fitness', marker='o')
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')
    plt.title('Convergence Curve: Fitness vs Generations (EHO)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()