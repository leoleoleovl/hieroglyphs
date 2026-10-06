from ultralytics import YOLO
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WEIGHTS    = os.path.join(BASE_DIR, "runs", "detect", "train-4", "weights", "best.pt")
DATA_YAML  = os.path.join(BASE_DIR, "hieroglyph_dataset", "data.yaml")
PIPELINE   = os.path.join(BASE_DIR, "pipeline.py")

if __name__ == "__main__":
    model = YOLO(WEIGHTS)

    model.train(
        data          = DATA_YAML,
        epochs        = 60,
        imgsz         = 640,
        batch         = 16,
        lr0           = 0.0005,
        warmup_epochs = 3,
        augment       = True,
        degrees       = 10,
        translate     = 0.1,
        scale         = 0.3,
        shear         = 5,
        perspective   = 0.0005,
        flipud        = 0.1,
        fliplr        = 0.5,
        mosaic        = 0.5,
        project       = os.path.join(BASE_DIR, "runs", "detect"),
        name          = "finetune-5",
        exist_ok      = False,
    )

    new_weights = os.path.join(BASE_DIR, "runs", "detect", "finetune-5", "weights", "best.pt")

    with open(PIPELINE, "r", encoding="utf-8") as f:
        src = f.read()

    updated = src.replace(
        "runs', 'detect', 'finetune-1', 'weights', 'best.pt'",
        "runs', 'detect', 'finetune-5', 'weights', 'best.pt'",
    )

    with open(PIPELINE, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"\nModel path in pipeline.py updated to: {new_weights}")
