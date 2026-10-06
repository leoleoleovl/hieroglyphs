"""
Test different preprocessing approaches to bridge the domain gap
between real stone carving photos and the model's training data.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

IMG_PATH   = r"C:\Users\admin\Downloads\Hieroglyphs-temple-Ombos-Egypt.png"
MODEL_PATH = r"C:\Users\admin\Desktop\hieroglyph_app\runs\detect\finetune-5\weights\best.pt"
OUT_DIR    = Path(r"C:\Users\admin\Desktop\hieroglyph_app\preproc_results")
OUT_DIR.mkdir(exist_ok=True)

model = YOLO(MODEL_PATH)
img   = cv2.imread(IMG_PATH)
gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def run(name, proc):
    bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR) if proc.ndim == 2 else proc
    cv2.imwrite(str(OUT_DIR / f"{name}_input.jpg"), bgr)
    r = model.predict(bgr, conf=0.01, iou=0.3, imgsz=1280, verbose=False)
    n = len(r[0].boxes)
    print(f"{name:25s}: {n} detections", end="")
    if n:
        confs = [b.conf.item() for b in r[0].boxes]
        print(f"  (max conf {max(confs):.3f})", end="")
        r[0].save(str(OUT_DIR / f"{name}_result.jpg"))
    print()
    return n

# 1. Adaptive threshold — turns carved shadows into black marks on white
adapt = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
run("1_adaptive_thresh", adapt)

# 2. Inverted adaptive threshold — for raised-relief where symbols are lighter
adapt_inv = cv2.bitwise_not(adapt)
run("2_adaptive_inv", adapt_inv)

# 3. Canny edges
blur  = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(blur, 20, 60)
edges_on_white = cv2.bitwise_not(edges)   # black edges on white (like training data)
run("3_canny_on_white", edges_on_white)

# 4. Difference of Gaussians (pencil-sketch style)
g1  = cv2.GaussianBlur(gray, (5, 5), 1)
g2  = cv2.GaussianBlur(gray, (21, 21), 5)
dog = np.clip(g1.astype(int) - g2.astype(int) + 128, 0, 255).astype(np.uint8)
run("4_dog_sketch", dog)

# 5. Aggressive CLAHE + invert (sunken-relief carvings become darker)
clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
cl    = clahe.apply(gray)
run("5_clahe_strong", cl)
run("5_clahe_strong_inv", cv2.bitwise_not(cl))

# 6. Local contrast (highlight areas that differ from surroundings — carved edges)
blurred  = cv2.GaussianBlur(gray, (31, 31), 0)
local_c  = np.clip(gray.astype(int) - blurred.astype(int) + 128, 0, 255).astype(np.uint8)
_, lc_bin = cv2.threshold(local_c, 140, 255, cv2.THRESH_BINARY_INV)
run("6_local_contrast", lc_bin)

print(f"\nPreprocessed images saved to: {OUT_DIR}")
print("Open that folder to see what each method looks like.")
