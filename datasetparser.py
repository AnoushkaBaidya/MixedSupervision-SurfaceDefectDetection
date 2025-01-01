import os
import shutil

# This script copies a specified number of files from categorized subdirectories (e.g., "good", "bad", "masks")
# in a source dataset directory to corresponding subdirectories in a destination directory.

# Define the paths for the dataset
source_directory = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/14_oct"  # Directory containing the original dataset
destination_directory = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/MyTrainDataset"  # Target directory for selected images

# Categories and number of files to copy from each
categories = {
    "good": 100,  # Copy 100 images from the "good" category
    "bad": 100,   # Copy 100 images from the "bad" category
    "masks": 100  # Copy 100 images from the "masks" category
}

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Loop through each category and process the files
for category, count in categories.items():
    source_path = os.path.join(source_directory, category)  # Path to the source category directory
    destination_path = os.path.join(destination_directory, category)  # Path to the destination category directory

    # Create the category folder in the destination directory if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)

    # List all files in the source directory for the current category
    files = os.listdir(source_path)
    
    # Select only the required number of files
    selected_files = files[:count]  # Take the first `count` files from the list

    # Copy the selected files to the destination directory
    for file in selected_files:
        shutil.copy(os.path.join(source_path, file), os.path.join(destination_path, file))  # Copy each file

# Print completion message
print(f"Selected images have been copied to the '{destination_directory}' directory.")
