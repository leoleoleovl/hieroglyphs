"""
Background augmentation: composite clean training symbols onto real stone
backgrounds from the HLA dataset with a carving effect, then save as new
training images. Run once before retraining.
"""
import cv2
import numpy as np
import random
import shutil
from pathlib import Path

HLA_DIR     = Path(r"C:\Users\admin\.cache\kagglehub\datasets\ahmedelkelany\egyptian-hieroglyphic-layout-analysis\versions\2")
DATASET_DIR = Path(r"C:\Users\admin\Desktop\hieroglyph_app\hieroglyph_dataset")
PREVIEW_DIR = Path(r"C:\Users\admin\Desktop\hieroglyph_app\aug_preview")
OUTPUT_SIZE = 640
AUGMENTS_PER_IMAGE = 2   # synthetic copies per original training image

# ── Load HLA background paths ─────────────────────────────────────────────────
BG_PATHS = (
    list(HLA_DIR.glob("*.jpg")) + list(HLA_DIR.glob("*.JPG")) +
    list(HLA_DIR.glob("*.png")) + list(HLA_DIR.glob("*.PNG"))
)
print(f"Found {len(BG_PATHS)} HLA background images")


def random_stone_crop(size=OUTPUT_SIZE):
    """Return a random size×size crop from a random HLA background."""
    for _ in range(20):
        bg = cv2.imread(str(random.choice(BG_PATHS)))
        if bg is None:
            continue
        h, w = bg.shape[:2]
        if h < size or w < size:
            bg = cv2.resize(bg, (max(w, size), max(h, size)))
            h, w = bg.shape[:2]
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        return bg[y:y+size, x:x+size].copy()
    return None


def carve_all_onto_stone(sym_bgr, lbl_lines, size=OUTPUT_SIZE):
    """
    Composite ALL symbols from one training image onto a stone background,
    preserving their relative layout, scaled to fit.

    sym_bgr   : original training image (BGR)
    lbl_lines : list of YOLO label strings  "class_id cx cy w h"

    Returns (composite_bgr, list_of_label_strings) or (None, None).
    """
    ih, iw = sym_bgr.shape[:2]

    # Parse all valid annotations
    valid = []
    for line in lbl_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = parts[0]
        cx_n, cy_n, bw_n, bh_n = map(float, parts[1:5])
        valid.append((class_id, cx_n, cy_n, bw_n, bh_n))

    if not valid:
        return None, None

    # ── Compute bounding box of the entire symbol group ──────────────────────
    xs1 = [cx - bw / 2 for _, cx, cy, bw, bh in valid]
    ys1 = [cy - bh / 2 for _, cx, cy, bw, bh in valid]
    xs2 = [cx + bw / 2 for _, cx, cy, bw, bh in valid]
    ys2 = [cy + bh / 2 for _, cx, cy, bw, bh in valid]

    g_x1, g_y1 = min(xs1), min(ys1)
    g_x2, g_y2 = max(xs2), max(ys2)
    g_w = max(g_x2 - g_x1, 1e-6)
    g_h = max(g_y2 - g_y1, 1e-6)

    # ── Scale the group so it fills 40-70 % of the output canvas ────────────
    fill = random.uniform(0.40, 0.70)
    pix_scale = min(fill * size / (g_w * iw), fill * size / (g_h * ih))

    group_pix_w = g_w * iw * pix_scale
    group_pix_h = g_h * ih * pix_scale

    # Random placement offset (keep group fully inside canvas)
    ox = random.randint(0, max(0, size - int(group_pix_w) - 1))
    oy = random.randint(0, max(0, size - int(group_pix_h) - 1))

    # ── Get stone background ─────────────────────────────────────────────────
    stone = random_stone_crop(size)
    if stone is None:
        return None, None

    new_labels = []

    for class_id, cx_n, cy_n, bw_n, bh_n in valid:
        # Pixel crop in original image
        x1 = int((cx_n - bw_n / 2) * iw)
        y1 = int((cy_n - bh_n / 2) * ih)
        x2 = int((cx_n + bw_n / 2) * iw)
        y2 = int((cy_n + bh_n / 2) * ih)

        pad = 3
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(iw, x2 + pad), min(ih, y2 + pad)

        sym_crop = sym_bgr[y1:y2, x1:x2]
        if sym_crop.size == 0:
            continue

        nsw = max(8, int((x2 - x1) * pix_scale))
        nsh = max(8, int((y2 - y1) * pix_scale))
        sym_s = cv2.resize(sym_crop, (nsw, nsh))

        # Position on stone, relative to group top-left corner
        rel_x = (cx_n - bw_n / 2 - g_x1) * iw * pix_scale
        rel_y = (cy_n - bh_n / 2 - g_y1) * ih * pix_scale
        px = int(ox + rel_x)
        py = int(oy + rel_y)

        # Skip if symbol would go out of canvas
        if px + nsw > size or py + nsh > size or px < 0 or py < 0:
            continue

        # ── Carving mask: OTSU on inverted gray ──────────────────────────────
        gray = cv2.cvtColor(sym_s, cv2.COLOR_BGR2GRAY)
        _, bw_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Soften edges to simulate stone wear
        blur_k = max(3, (min(nsw, nsh) // 8) | 1)
        mask = cv2.GaussianBlur(bw_mask.astype(np.float32) / 255.0,
                                (blur_k, blur_k), 2)

        # ── Apply carving: deeper darkening than before ───────────────────────
        depth  = random.uniform(0.40, 0.70)
        region = stone[py:py+nsh, px:px+nsw].astype(np.float32)
        mask3  = np.stack([mask] * 3, axis=2)
        stone[py:py+nsh, px:px+nsw] = np.clip(
            region * (1 - depth * mask3), 0, 255
        ).astype(np.uint8)

        # ── New normalised bounding box ───────────────────────────────────────
        ncx = (px + nsw / 2) / size
        ncy = (py + nsh / 2) / size
        nbw = nsw / size
        nbh = nsh / size
        new_labels.append(f"{class_id} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")

    if not new_labels:
        return None, None

    return stone, new_labels


def process_split(split, preview_only=False, preview_n=8):
    img_dir = DATASET_DIR / "images" / split
    lbl_dir = DATASET_DIR / "labels" / split

    img_paths = (
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpeg"))
    )
    # Skip already-augmented files so re-runs are safe
    img_paths = [p for p in img_paths if not p.stem.startswith("bgaug_")]
    random.shuffle(img_paths)

    if preview_only:
        img_paths = img_paths[:preview_n]

    done = 0
    for img_path in img_paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        lines = [l for l in lbl_path.read_text(encoding="utf-8").strip().splitlines() if l]
        if not lines:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for aug_i in range(AUGMENTS_PER_IMAGE):
            composite, new_lbls = carve_all_onto_stone(img, lines)
            if composite is None:
                continue

            stem = f"bgaug_{img_path.stem}_{aug_i}"

            if preview_only:
                PREVIEW_DIR.mkdir(exist_ok=True)
                cv2.imwrite(str(PREVIEW_DIR / f"{stem}.jpg"), composite)
            else:
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), composite)
                (lbl_dir / f"{stem}.txt").write_text(
                    "\n".join(new_lbls), encoding="utf-8"
                )

            done += 1

        if not preview_only and done % 500 == 0:
            print(f"  [{split}] {done} augmented images written…")

    return done


# ── STEP 1: Generate preview samples ─────────────────────────────────────────
print("\n--- Generating 8 preview samples (check aug_preview/ folder) ---")
process_split("train", preview_only=True, preview_n=8)
print(f"Preview saved to: {PREVIEW_DIR}")
print("Open aug_preview/ and check the images look like stone carvings.")
print("If they look good, re-run with: python augment_backgrounds.py --go\n")

# ── STEP 2: Full augmentation (only runs when --go flag is passed) ────────────
import sys
if "--go" in sys.argv:
    print("--- Running full augmentation ---")
    for split in ["train", "val"]:
        n = process_split(split)
        print(f"  {split}: {n} augmented images added")

    # Delete YOLO dataset cache so it rescans on next training run
    for cache in DATASET_DIR.rglob("*.cache"):
        cache.unlink()
        print(f"  Deleted cache: {cache}")

    print("\nDone. Now run: python finetune.py")
