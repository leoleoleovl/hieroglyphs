import os
import glob
import cv2
import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import shutil
# ***********************************************************
#                    PATHS AND CONFIG
# ***********************************************************

 # Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
   
   # Configure paths - users should update these to their dataset location
DATASET_PATH = os.path.join(BASE_DIR, "dataset")  # or provide your dataset path
PREPROCESSED_PATH = os.path.join(BASE_DIR, "preprocessed_data")

#  preprocessed directories
os.makedirs(PREPROCESSED_PATH, exist_ok=True)
for split in ["train", "valid", "test"]:
    os.makedirs(f"{PREPROCESSED_PATH}/{split}/images", exist_ok=True)
    os.makedirs(f"{PREPROCESSED_PATH}/{split}/labels", exist_ok=True)

TARGET_SIZE = (640, 640)  # 640x640  input size


# *******************************************************
#             1. PREPROCESSING PIPELINE
# *****************************************************

def preprocess_dataset():
    print("Starting image preprocessing...")

    for split in ["train", "valid", "test"]:
        img_dir = f"{DATASET_PATH}/{split}/images"
        lbl_dir = f"{DATASET_PATH}/{split}/labels"

        out_img_dir = f"{PREPROCESSED_PATH}/{split}/images"
        out_lbl_dir = f"{PREPROCESSED_PATH}/{split}/labels"

        for img_file in os.listdir(img_dir):

            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(lbl_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

            # Copy labels as-is
            if os.path.exists(label_path):
                   shutil.copy2(label_path, out_lbl_dir)

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Warning] Skipping corrupted image: {img_file}")
                continue

            # 1. Resize
            img = cv2.resize(img, TARGET_SIZE)

            # 2. Normalize
            img = img / 255.0

            # 3. Denoise
            img = cv2.fastNlMeansDenoisingColored((img * 255).astype("uint8"), None, 10, 10, 7, 21)

            # 4. Contrast enhancement 
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge((l, a, b))
            img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            # Save preprocessed image
            cv2.imwrite(f"{out_img_dir}/{img_file}", img)

    print("Preprocessing complete.")
if __name__ == "__main__":

    preprocess_dataset()
