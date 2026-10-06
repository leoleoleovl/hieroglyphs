import os
import shutil
import csv
from pathlib import Path
import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
NEW_DATASET   = Path(r"C:\Users\admin\Downloads\archive")
EXISTING      = Path(r"C:\Users\admin\Desktop\hieroglyph_app\hieroglyph_dataset")
YAML_PATH     = EXISTING / "data.yaml"

# ── Load class name → ID mapping from existing data.yaml ────────────────────
with open(YAML_PATH) as f:
    cfg = yaml.safe_load(f)

# cfg["names"] is {0: "100", 1: "Among", ...}  → invert to {"100": 0, ...}
CLASS_MAP = {v: k for k, v in cfg["names"].items()}
print(f"Loaded {len(CLASS_MAP)} classes from data.yaml")

# ── Merge mapping: new split → existing split ────────────────────────────────
# train → train,  valid → val,  test → train (extra training data)
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test":  "train",
}

skipped_classes = set()
total_images = 0
total_labels = 0

for new_split, existing_split in SPLIT_MAP.items():
    csv_path = NEW_DATASET / new_split / "_annotations.csv"
    img_src  = NEW_DATASET / new_split
    img_dst  = EXISTING / "images" / existing_split
    lbl_dst  = EXISTING / "labels" / existing_split

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"  No annotations found for split '{new_split}', skipping.")
        continue

    # Group annotations by filename
    annotations: dict[str, list] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            if fname not in annotations:
                annotations[fname] = []
            annotations[fname].append(row)

    split_images = 0
    split_labels = 0

    for fname, rows in annotations.items():
        src_img = img_src / fname
        if not src_img.exists():
            continue

        w = float(rows[0]["width"])
        h = float(rows[0]["height"])
        if w == 0 or h == 0:
            continue

        yolo_lines = []
        for row in rows:
            cls = row["class"]
            if cls not in CLASS_MAP:
                skipped_classes.add(cls)
                continue

            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])

            cx = (xmin + xmax) / 2 / w
            cy = (ymin + ymax) / 2 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            yolo_lines.append(f"{CLASS_MAP[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            continue

        # Copy image (prefix with split name to avoid filename collisions)
        dst_img = img_dst / f"new_{fname}"
        shutil.copy2(src_img, dst_img)

        # Write YOLO label file
        stem    = Path(fname).stem
        lbl_out = lbl_dst / f"new_{stem}.txt"
        lbl_out.write_text("\n".join(yolo_lines))

        split_images += 1
        split_labels += len(yolo_lines)

    print(f"  {new_split:6s} → {existing_split:6s}:  {split_images} images,  {split_labels} bounding boxes")
    total_images += split_images
    total_labels += split_labels

print(f"\nDone.  Added {total_images} images and {total_labels} bounding boxes total.")

if skipped_classes:
    print(f"Skipped unknown classes: {skipped_classes}")
