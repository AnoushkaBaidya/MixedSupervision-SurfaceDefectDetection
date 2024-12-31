import os
import shutil

# Define the paths for the dataset
source_directory = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/14_oct"
destination_directory = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/MyTrainDataset"  # This is where the selected images will go

# Subdirectories
categories = {
    "good": 100,
    "bad": 100,
    "masks": 100
}

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

for category, count in categories.items():
    source_path = os.path.join(source_directory, category)
    destination_path = os.path.join(destination_directory, category)
    
    # Create category folder in the destination directory
    os.makedirs(destination_path, exist_ok=True)

    # List all files in the source directory for the category
    files = os.listdir(source_path)
    
    # Select only the required number of files
    selected_files = files[:count]

    # Copy the selected files to the destination directory
    for file in selected_files:
        shutil.copy(os.path.join(source_path, file), os.path.join(destination_path, file))

print(f"Selected images have been copied to the '{destination_directory}' directory.")
