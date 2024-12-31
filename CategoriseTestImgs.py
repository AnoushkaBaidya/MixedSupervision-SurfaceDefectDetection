import os
import shutil

# Paths to the original directories
test_dir = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/TestImages/8Nov"
good_dir = os.path.join(test_dir, "good")
bad_dir = os.path.join(test_dir, "bad")

# Path to the new directory
new_dir = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/TestImages/8_Nov_Test_Dataset"
os.makedirs(new_dir, exist_ok=True)

# Function to rename and move images
def rename_and_move_images(src_dir, dst_dir, prefix):
    if not os.path.exists(src_dir):
        print(f"Directory {src_dir} does not exist. Skipping.")
        return

    images = os.listdir(src_dir)
    for idx, img_name in enumerate(images, start=1):
        # Full paths to the source and destination
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, f"{prefix}{idx}.png")  # Change extension if not .png

        # Move and rename the file
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)

# Rename and move good images
rename_and_move_images(good_dir, new_dir, "good")

# Rename and move bad images
rename_and_move_images(bad_dir, new_dir, "bad")

print("Renaming and moving completed!")
