import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Function to extract LBP features
def extract_lbp_features(image_path, P=8, R=1):
    """
    Extract LBP features from an image.
    :param image_path: Path to the image file.
    :param P: Number of circularly symmetric neighbor points.
    :param R: Radius of the circle.
    :return: Flattened LBP feature vector.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}. Check if the file is a valid image.")
    
    # Resize the image to a fixed size (e.g., 128x128)
    image = cv2.resize(image, (128, 128))
    
    # Compute LBP features
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    
    # Flatten the LBP matrix into a feature vector
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize the histogram
    
    return lbp_hist

# Function to perform EHO feature selection
def eho_feature_selection(features, fitness_function, num_iterations=50, population_size=20):
    """
    Perform feature selection using Elephant Herding Optimization (EHO).
    :param features: Input feature matrix.
    :param fitness_function: Function to evaluate fitness of a feature subset.
    :param num_iterations: Number of iterations for EHO.
    :param population_size: Size of the population.
    :return: Binary mask indicating selected features.
    """
    # Initialize population (binary vectors representing feature subsets)
    num_features = features.shape[1]
    population = np.random.randint(2, size=(population_size, num_features))
    
    # Iterate through generations
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        # Evaluate fitness of each individual
        fitness_scores = []
        for individual in population:
            selected_features = features[:, individual == 1]  # Select features
            if selected_features.shape[1] == 0:  # Skip if no features are selected
                fitness_scores.append(0)
                continue
            fitness_scores.append(fitness_function(selected_features))
        fitness_scores = np.array(fitness_scores)
        
        # Select the best individuals (simplified EHO logic)
        best_indices = np.argsort(fitness_scores)[-population_size//2:]
        population = population[best_indices]
        
        # Generate new individuals (crossover and mutation)
        new_population = []
        for i in range(population_size):
            # Select two parents randomly from the best individuals
            parent1, parent2 = np.random.choice(len(best_indices), 2, replace=False)
            parent1 = population[parent1]
            parent2 = population[parent2]
            
            # Perform crossover (bitwise OR operation)
            child = parent1 | parent2
            
            # Perform mutation (flip a random bit with 10% probability)
            if np.random.rand() < 0.1:
                mutation_index = np.random.randint(num_features)
                child[mutation_index] = 1 - child[mutation_index]
            
            new_population.append(child)
        
        # Update the population
        population = np.array(new_population)
    
    # Return the best feature subset
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual == 1  # Return a boolean mask for selected features

# Function to visualize LBP features for a sample image
def visualize_lbp_features(image_path, P=8, R=1):
    """
    Visualize the original image and its LBP histogram.
    :param image_path: Path to the image file.
    :param P: Number of circularly symmetric neighbor points.
    :param R: Radius of the circle.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}. Check if the file is a valid image.")
    
    # Resize the image to a fixed size (e.g., 128x128)
    image = cv2.resize(image, (128, 128))
    
    # Compute LBP features
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    
    # Compute LBP histogram
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize the histogram
    
    # Plot the original image and LBP histogram
    plt.figure(figsize=(10, 5))
    
    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    
    # Plot the LBP histogram
    plt.subplot(1, 2, 2)
    plt.bar(range(len(lbp_hist)), lbp_hist)
    plt.title("LBP Histogram")
    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Frequency")
    
    plt.tight_layout()
    plt.show()

# Function to compare full and selected LBP features
def compare_full_vs_selected_features(image_path, selected_features_mask):
    """
    Compare the full LBP histogram with the selected LBP histogram.
    :param image_path: Path to the image file.
    :param selected_features_mask: Binary mask indicating selected features.
    """
    # Extract full LBP features
    lbp_features = extract_lbp_features(image_path)
    
    # Extract selected LBP features
    selected_lbp_features = lbp_features[selected_features_mask]
    
    # Plot the full and selected LBP histograms
    plt.figure(figsize=(10, 5))
    
    # Plot the full LBP histogram
    plt.subplot(1, 2, 1)
    plt.bar(range(len(lbp_features)), lbp_features)
    plt.title("Full LBP Histogram")
    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Frequency")
    
    # Plot the selected LBP histogram
    plt.subplot(1, 2, 2)
    plt.bar(range(len(selected_lbp_features)), selected_lbp_features)
    plt.title("Selected LBP Histogram")
    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Frequency")
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
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
    
    # Perform EHO feature selection (example template)
    print("Performing EHO feature selection...")
    def fitness_function(features):
        # Define a simple fitness function (e.g., accuracy from a classifier)
        clf = LogisticRegression()
        clf.fit(features, y_train)  # Use training data for feature selection
        return clf.score(features, y_train)  # Evaluate on training data
    
    # Perform EHO feature selection
    selected_features_mask = eho_feature_selection(X_train, fitness_function)
    print("Selected features mask:", selected_features_mask)
    
    # Visualize the selected features mask
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(selected_features_mask)), selected_features_mask)
    plt.title("Selected Features Mask")
    plt.xlabel("Feature Index")
    plt.ylabel("Selected (1) or Not Selected (0)")
    plt.show()
    
    # Visualize feature importance
    clf = LogisticRegression()
    clf.fit(X_train[:, selected_features_mask], y_train)
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(X_train[:, selected_features_mask].shape[1]), clf.coef_[0])
    plt.title("Feature Importance (Selected Features)")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance Score")
    plt.show()
    
    # Visualize LBP features for a sample image
    sample_image_path = "data/processed/healthy/healthy_002.jpg"  # Replace with a valid path
    try:
        visualize_lbp_features(sample_image_path)
    except Exception as e:
        print(f"Error visualizing LBP features for {sample_image_path}: {e}")
    
    # Compare full and selected LBP features for a sample image
    try:
        compare_full_vs_selected_features(sample_image_path, selected_features_mask)
    except Exception as e:
        print(f"Error comparing full vs. selected features for {sample_image_path}: {e}")

if __name__ == "__main__":
    main()