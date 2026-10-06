import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# 1. Setup Base and Target Directories
RAW_DATA_DIR = Path(r"C:\Users\admin\.cache\kagglehub\datasets\waleedumer\egyptian-hieroglyphics-datasets\versions\1")
OUTPUT_DIR = Path(r".\hieroglyph_dataset")

# Build target directory structure for YOLOv8
for split in ['train', 'val']:
    (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

print("Scanning XML metadata to index unique classes...")

# 2. Map Unique Class Labels Dynamically
unique_classes = set()
xml_files = list(RAW_DATA_DIR.rglob("*.xml"))

if not xml_files:
    print(f"Error: Couldn't locate any XML files in {RAW_DATA_DIR}. Check your path alignment.")
    exit(1)

for xml_path in xml_files:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            class_name = obj.find('name').text.strip()
            unique_classes.add(class_name)
    except Exception:
        continue

class_mapping = {name: idx for idx, name in enumerate(sorted(list(unique_classes)))}
print(f"Success: Indexed {len(class_mapping)} unique hieroglyph classes.")

# 3. Create Train/Validation Split (80% Train, 20% Val)
random.seed(42)  # Ensures split consistency every time you run it
random.shuffle(xml_files)
split_idx = int(len(xml_files) * 0.8)
train_files = xml_files[:split_idx]
val_files = xml_files[split_idx:]

def process_split(files, split_name):
    print(f"Processing {split_name} split ({len(files)} annotation sets)...")
    
    for xml_path in files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Resolve image filename directly from the XML meta-tags
            img_filename = root.find('filename').text.strip()
            img_source_path = xml_path.parent / img_filename
            
            # Safe fallback logic if extension casing or naming differs slightly
            if not img_source_path.exists():
                fallback_jpg = xml_path.with_suffix('.jpg')
                fallback_png = xml_path.with_suffix('.png')
                if fallback_jpg.exists():
                    img_source_path = fallback_jpg
                elif fallback_png.exists():
                    img_source_path = fallback_png
                else:
                    print(f"Skipping: Reference image {img_filename} missing for {xml_path.name}")
                    continue

            # Read spatial boundaries for normalization
            size = root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            
            if width == 0 or height == 0:
                continue

            # Mirror the naming format uniformly for target image/label pairs
            base_name = xml_path.stem
            img_target_path = OUTPUT_DIR / 'images' / split_name / f"{base_name}{img_source_path.suffix}"
            label_target_path = OUTPUT_DIR / 'labels' / split_name / f"{base_name}.txt"
            
            # Parse bounding boxes into relative YOLO coordinates
            yolo_labels = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text.strip()
                class_id = class_mapping[class_name]
                
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # Transform to relative center point, width, and height (0.0 to 1.0)
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
            # If coordinates are valid, isolate into destination directory
            if yolo_labels:
                shutil.copy(img_source_path, img_target_path)
                with open(label_target_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                    
        except Exception as e:
            print(f"Error handling file {xml_path.name}: {e}")

# Process datasets
process_split(train_files, 'train')
process_split(val_files, 'val')

# 4. Automate configuration file generation
yaml_path = OUTPUT_DIR / 'data.yaml'
with open(yaml_path, 'w') as f:
    f.write(f"path: {OUTPUT_DIR.resolve()}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write("names:\n")
    for name, idx in class_mapping.items():
        f.write(f"  {idx}: {name}\n")

print(f"\nProcessing Complete. Organized architecture built at: {OUTPUT_DIR.resolve()}")
print(f"Configuration profile written to: {yaml_path.name}")