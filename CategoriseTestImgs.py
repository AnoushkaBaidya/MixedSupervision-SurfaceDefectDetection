import os
import shutil

# Paths to the original directories
# Define the main test directory containing subdirectories for "good" and "bad" images
test_dir = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/TestImages/8Nov"
good_dir = os.path.join(test_dir, "good")  # Subdirectory for "good" images
bad_dir = os.path.join(test_dir, "bad")  # Subdirectory for "bad" images

# Path to the new directory where images will be renamed and consolidated
new_dir = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/TestImages/8_Nov_Test_Dataset"
os.makedirs(new_dir, exist_ok=True)  # Ensure the new directory exists; create it if necessary

# Function to rename and move images from source to destination
# Parameters:
# - src_dir: Source directory containing images to be moved
# - dst_dir: Destination directory where images will be moved
# - prefix: Prefix to be added to the renamed files
def rename_and_move_images(src_dir, dst_dir, prefix):
    # Check if the source directory exists; if not, print a warning and skip
    if not os.path.exists(src_dir):
        print(f"Directory {src_dir} does not exist. Skipping.")
        return

    # Get a list of all images in the source directory
    images = os.listdir(src_dir)
    for idx, img_name in enumerate(images, start=1):
        # Full paths to the source and destination files
        src_path = os.path.join(src_dir, img_name)  # Source file path
        dst_path = os.path.join(dst_dir, f"{prefix}{idx}.png")  # Destination file path with renamed file

        # Move and rename the file only if it is a valid file (not a directory)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)  # Copy the file to the destination directory

# Rename and move "good" images
# Add "good" prefix to images in the "good" directory and copy them to the new directory
rename_and_move_images(good_dir, new_dir, "good")

# Rename and move "bad" images
# Add "bad" prefix to images in the "bad" directory and copy them to the new directory
rename_and_move_images(bad_dir, new_dir, "bad")

# Print completion message
print("Renaming and moving completed!")
