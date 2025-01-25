import cv2
import numpy as np
from skimage.feature import local_binary_pattern

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
    
    # Check if the image was loaded successfully
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