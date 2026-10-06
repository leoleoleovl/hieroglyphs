"""
Tile the real photo into overlapping 640x640 patches, preprocess each,
run detection, then merge all boxes back to original coordinates.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

IMG_PATH   = r"C:\Users\admin\Downloads\Hieroglyphs-temple-Ombos-Egypt.png"
MODEL_PATH = r"C:\Users\admin\Desktop\hieroglyph_app\runs\detect\finetune-5\weights\best.pt"
OUT_DIR    = Path(r"C:\Users\admin\Desktop\hieroglyph_app\preproc_results")
OUT_DIR.mkdir(exist_ok=True)

TILE   = 320   # smaller crop → symbols fill more of the 640 model input
STRIDE = 160   # 50 % overlap
CONF   = 0.01

model = YOLO(MODEL_PATH)
img   = cv2.imread(IMG_PATH)
H, W  = img.shape[:2]


def preprocess(patch_gray):
    """Best method from preproc_test: aggressive CLAHE then invert."""
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
    cl    = clahe.apply(patch_gray)
    return cv2.bitwise_not(cl)


all_boxes  = []   # [x1,y1,x2,y2,conf,cls]
gray_full  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ys = list(range(0, H - TILE + 1, STRIDE)) + ([H - TILE] if H > TILE else [])
xs = list(range(0, W - TILE + 1, STRIDE)) + ([W - TILE] if W > TILE else [])

total_tiles = len(set(ys)) * len(set(xs))
print(f"Image: {W}×{H}   Tiles: {total_tiles}")

for y in set(ys):
    for x in set(xs):
        patch_gray = gray_full[y:y+TILE, x:x+TILE]
        patch_proc = preprocess(patch_gray)
        patch_bgr  = cv2.cvtColor(patch_proc, cv2.COLOR_GRAY2BGR)

        r = model.predict(patch_bgr, conf=CONF, iou=0.3, imgsz=640, verbose=False)
        for b in r[0].boxes:
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            all_boxes.append([
                x + bx1, y + by1, x + bx2, y + by2,
                b.conf.item(), int(b.cls)
            ])

print(f"Raw detections before NMS: {len(all_boxes)}")

if all_boxes:
    boxes_np  = np.array(all_boxes)
    # Simple NMS across tiles
    from torchvision.ops import nms
    import torch
    keep = nms(
        torch.tensor(boxes_np[:, :4], dtype=torch.float32),
        torch.tensor(boxes_np[:, 4],  dtype=torch.float32),
        iou_threshold=0.4,
    ).numpy()
    kept = boxes_np[keep]
    print(f"After NMS: {len(kept)} detections")

    # Draw on original image
    vis = img.copy()
    names = model.names
    for x1, y1, x2, y2, conf, cls in kept:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{names[int(cls)]} {conf:.2f}"
        cv2.putText(vis, label, (int(x1), max(int(y1)-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out_path = str(OUT_DIR / "tiled_result.jpg")
    cv2.imwrite(out_path, vis)
    print(f"Result saved to: {out_path}")

    for x1, y1, x2, y2, conf, cls in sorted(kept, key=lambda r: -r[4])[:20]:
        print(f"  {names[int(cls)]:20s}  conf={conf:.3f}")
else:
    print("No detections at all.")
