import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
# Model path - looks for weights in runs/segment/hieroglyph_seg/weights/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'segment', 'hieroglyph_seg', 'weights', 'best.pt')
IMG_SIZE = 640
# ----------------------------------------
class HieroglyphGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hieroglyph Detection & Segmentation")

        # Load model once
        self.model = YOLO(MODEL_PATH)
        print("!!! MODEL CLASSES ARE DIRECTLY TARGETING:", self.model.names)

        # UI Elements
        self.btn_load = tk.Button(root, text="Load Image", command=self.load_image)
        self.btn_load.pack(pady=10)

        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.image_path = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )

        if not self.image_path:
            return

        self.run_inference()

    def run_inference(self):
        # Read image
        image = cv2.imread(self.image_path)

        # --- NEW PREPROCESSING ADDED HERE ---
        # 1. Convert to grayscale so we can isolate the textures
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply CLAHE contrast enhancement to pop out faint outlines
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # 3. Convert enhanced image to RGB channel format for YOLO
        model_input = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        # ------------------------------------

        # Convert original image to RGB so your GUI background looks normal
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Inference: We feed the enhanced image here and bump conf back up to 0.25
        results = self.model(model_input, imgsz=IMG_SIZE, conf=0.20, iou=0.4)[0]

        output = image.copy()

        # Draw masks
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            for mask in masks:
                color = np.array([0, 255, 0], dtype=np.uint8)
                mask = cv2.resize(mask, (output.shape[1], output.shape[0]))
                output[mask > 0.5] = (
                    output[mask > 0.5] * 0.5 + color * 0.5
                )

        # Draw bounding boxes
        if results.boxes is not None:
            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Convert to Tk image
        img = Image.fromarray(output)
        img.thumbnail((800, 800))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.tk_img)

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = HieroglyphGUI(root)
    root.mainloop()

