import os
import shutil
from tqdm import tqdm

# Define source and target directories
source_dir = "data/raw/PlantVillage"  # Update this path if necessary
target_dir = "data/processed"
os.makedirs(target_dir, exist_ok=True)

# Create directories for healthy and diseased images
healthy_dir = os.path.join(target_dir, "healthy")
diseased_dir = os.path.join(target_dir, "diseased")
os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(diseased_dir, exist_ok=True)

# Iterate through each class folder in the dataset
for folder in tqdm(os.listdir(source_dir)):
    folder_path = os.path.join(source_dir, folder)
    
    # Skip if it's not a directory
    if not os.path.isdir(folder_path):
        continue

    # Determine target folder based on class name
    if "healthy" in folder.lower():
        dest = healthy_dir
    else:
        dest = diseased_dir

    # Copy all files from the current folder to the target directory
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):  # Ensure it's a file
            shutil.copy(file_path, dest)

print("Dataset organization complete!")
