import cv2
import os

file_path = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/datasets/KSDD2/train/10000.png"
if os.path.exists(file_path):
    img = cv2.imread(file_path)
    if img is None:
        print("OpenCV cannot read the file. Check the file format or permissions.")
    else:
        print("File is accessible and readable.")
else:
    print("File path does not exist.")
