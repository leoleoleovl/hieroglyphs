import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'finetune-5', 'weights', 'best.pt')

# Mapping from model class names (as in data.yaml) to Egyptological data.
# phonetic: standard transliteration value; meaning: semantic gloss; gardiner: sign-list code.
HIEROGLYPH_LOOKUP = {
    "100":                   {"phonetic": "—",       "meaning": "hundred (numeral)",                  "gardiner": "Z1×100"},
    "Among":                 {"phonetic": "m-ẖnw",   "meaning": "among, within",                      "gardiner": "—"},
    "Angry":                 {"phonetic": "—",       "meaning": "anger, fury (determinative)",         "gardiner": "—"},
    "Ankh":                  {"phonetic": "ʿnḫ",     "meaning": "life",                               "gardiner": "S34"},
    "Aroura":                {"phonetic": "st.t",    "meaning": "aroura (unit of land ~0.27 ha)",      "gardiner": "—"},
    "At":                    {"phonetic": "m",       "meaning": "at, in, by",                          "gardiner": "—"},
    "Bad_Thinking":          {"phonetic": "—",       "meaning": "evil intent, bad thought",            "gardiner": "—"},
    "Bandage":               {"phonetic": "mnḫ",     "meaning": "bandage; excellent",                  "gardiner": "S28"},
    "Bee":                   {"phonetic": "bit",     "meaning": "bee; Lower Egypt (royal title)",      "gardiner": "L2"},
    "Belongs":               {"phonetic": "n",       "meaning": "belongs to, of",                      "gardiner": "—"},
    "Birth":                 {"phonetic": "ms",      "meaning": "birth, to give birth",                "gardiner": "—"},
    "Board_Game":            {"phonetic": "mn",      "meaning": "senet board; to remain, endure",      "gardiner": "Y5"},
    "Book":                  {"phonetic": "mḏȝt",    "meaning": "book, papyrus scroll",                "gardiner": "Y1"},
    "Boy":                   {"phonetic": "ẖrd",     "meaning": "boy, child",                          "gardiner": "—"},
    "Branch":                {"phonetic": "ḫt",      "meaning": "branch, wood, tree",                  "gardiner": "M3"},
    "Bread":                 {"phonetic": "t",       "meaning": "bread (phonetic: t)",                 "gardiner": "X1"},
    "Brewer":                {"phonetic": "—",       "meaning": "brewer (occupational determinative)", "gardiner": "—"},
    "Builder":               {"phonetic": "—",       "meaning": "builder, craftsman",                  "gardiner": "—"},
    "Bury":                  {"phonetic": "—",       "meaning": "to bury, interment",                  "gardiner": "—"},
    "Canal":                 {"phonetic": "mr",      "meaning": "canal; to love",                      "gardiner": "N36"},
    "Cloth_on_Pole":         {"phonetic": "nṯr",     "meaning": "god, divine",                         "gardiner": "R8"},
    "Cobra":                 {"phonetic": "wȝḏ",     "meaning": "cobra; fresh, green",                 "gardiner": "I13"},
    "Composite_Bow":         {"phonetic": "—",       "meaning": "composite bow (weapon)",              "gardiner": "T10"},
    "Cooked":                {"phonetic": "—",       "meaning": "cooked food (determinative)",          "gardiner": "—"},
    "Corpse":                {"phonetic": "ẖȝt",     "meaning": "corpse, body",                        "gardiner": "—"},
    "Dessert":               {"phonetic": "dšrt",    "meaning": "desert, red land",                    "gardiner": "N25"},
    "Divide":                {"phonetic": "—",       "meaning": "to divide, split",                    "gardiner": "—"},
    "Duck":                  {"phonetic": "sȝ",      "meaning": "duck; son (of)",                      "gardiner": "G38"},
    "Elephant":              {"phonetic": "ȝbw",     "meaning": "elephant; Elephantine",               "gardiner": "E26"},
    "Enclosed_Mound":        {"phonetic": "—",       "meaning": "enclosed mound, primordial hill",     "gardiner": "—"},
    "Eye":                   {"phonetic": "ir",      "meaning": "eye; to do, make, see",               "gardiner": "D4"},
    "Fabric":                {"phonetic": "—",       "meaning": "fabric, cloth (determinative)",       "gardiner": "—"},
    "Face":                  {"phonetic": "ḥr",      "meaning": "face; upon, over",                    "gardiner": "D2"},
    "Falcon":                {"phonetic": "ḥr",      "meaning": "falcon; Horus",                       "gardiner": "G5"},
    "Fingre":                {"phonetic": "ḏbʿ",     "meaning": "finger",                              "gardiner": "D50"},
    "Fish":                  {"phonetic": "—",       "meaning": "fish (determinative)",                "gardiner": "K1"},
    "Flail":                 {"phonetic": "nḫȝḫȝ",   "meaning": "flail (royal insignia)",              "gardiner": "S45"},
    "Folded_Cloth":          {"phonetic": "s",       "meaning": "folded cloth (phonetic: s)",          "gardiner": "S29"},
    "Foot":                  {"phonetic": "b",       "meaning": "foot (phonetic: b)",                  "gardiner": "D58"},
    "Galena":                {"phonetic": "msdmt",   "meaning": "galena, kohl (eye paint)",            "gardiner": "—"},
    "Giraffe":               {"phonetic": "sr",      "meaning": "giraffe; to foretell",                "gardiner": "E27"},
    "He":                    {"phonetic": "sw",      "meaning": "he, him (3rd person masc.)",          "gardiner": "—"},
    "Head":                  {"phonetic": "tp",      "meaning": "head; upon, first",                   "gardiner": "D1"},
    "Her":                   {"phonetic": "sy",      "meaning": "she, her (3rd person fem.)",          "gardiner": "—"},
    "Hit":                   {"phonetic": "—",       "meaning": "to hit, strike",                      "gardiner": "—"},
    "Horn":                  {"phonetic": "db",      "meaning": "horn",                                "gardiner": "—"},
    "King":                  {"phonetic": "nsw",     "meaning": "king, pharaoh",                       "gardiner": "—"},
    "Leg":                   {"phonetic": "rd",      "meaning": "leg; to grow",                        "gardiner": "D56"},
    "Length_Of_a_Human_Arm": {"phonetic": "mḥ",      "meaning": "cubit (unit of length ≈52 cm)",       "gardiner": "—"},
    "Life_Spirit":           {"phonetic": "kȝ",      "meaning": "ka, life-spirit, soul",               "gardiner": "D28"},
    "Limit":                 {"phonetic": "—",       "meaning": "limit, boundary",                     "gardiner": "—"},
    "Lion":                  {"phonetic": "rw",      "meaning": "lion",                                "gardiner": "E23"},
    "Lizard":                {"phonetic": "ȝš",      "meaning": "lizard",                              "gardiner": "I2"},
    "Loaf":                  {"phonetic": "t",       "meaning": "loaf of bread (phonetic: t)",         "gardiner": "X2"},
    "Loaf_Of_Bread":         {"phonetic": "t",       "meaning": "loaf of bread, offering",             "gardiner": "X2"},
    "Man":                   {"phonetic": "s",       "meaning": "man (seated determinative)",          "gardiner": "A1"},
    "Mascot":                {"phonetic": "—",       "meaning": "mascot, amulet figure",               "gardiner": "—"},
    "Meet":                  {"phonetic": "gb",      "meaning": "to meet, encounter",                  "gardiner": "—"},
    "Mother":                {"phonetic": "mwt",     "meaning": "mother",                              "gardiner": "H6"},
    "Mouth":                 {"phonetic": "r",       "meaning": "mouth (phonetic: r)",                 "gardiner": "D21"},
    "Musical_Instrument":    {"phonetic": "—",       "meaning": "musical instrument (determinative)",  "gardiner": "—"},
    "Nile_Fish":             {"phonetic": "in",      "meaning": "Nile fish (tilapia); to bring",       "gardiner": "K1"},
    "Not":                   {"phonetic": "n",       "meaning": "not, negation",                       "gardiner": "—"},
    "Now":                   {"phonetic": "—",       "meaning": "now, at this time",                   "gardiner": "—"},
    "Nurse":                 {"phonetic": "mnʿt",    "meaning": "nurse, wet nurse",                    "gardiner": "—"},
    "Nursing":               {"phonetic": "—",       "meaning": "nursing, suckling",                   "gardiner": "—"},
    "Occur":                 {"phonetic": "ḫpr",     "meaning": "to occur, become, exist",             "gardiner": "—"},
    "One":                   {"phonetic": "wʿ",      "meaning": "one (numeral)",                       "gardiner": "Z1"},
    "Owl":                   {"phonetic": "m",       "meaning": "owl (phonetic: m)",                   "gardiner": "G17"},
    "Pair":                  {"phonetic": "—",       "meaning": "pair, two",                           "gardiner": "—"},
    "Papyrus_Scroll":        {"phonetic": "—",       "meaning": "papyrus scroll (writing det.)",       "gardiner": "Y1"},
    "Pool":                  {"phonetic": "š",       "meaning": "pool, lake, garden",                  "gardiner": "N37"},
    "QuailChick":            {"phonetic": "w",       "meaning": "quail chick (phonetic: w/u)",         "gardiner": "G43"},
    "Reed":                  {"phonetic": "i",       "meaning": "reed (phonetic: i/j)",                "gardiner": "M17"},
    "Ring":                  {"phonetic": "šn",      "meaning": "shen ring; to encircle",              "gardiner": "V9"},
    "Rope":                  {"phonetic": "šn",      "meaning": "rope coil; hundred",                  "gardiner": "V1"},
    "Ruler":                 {"phonetic": "—",       "meaning": "ruler, measuring rod",                "gardiner": "—"},
    "Sail":                  {"phonetic": "nfw",     "meaning": "sail; wind, breath",                  "gardiner": "P5"},
    "Sandal":                {"phonetic": "—",       "meaning": "sandal (determinative)",              "gardiner": "S33"},
    "Semen":                 {"phonetic": "mtwt",    "meaning": "semen, seed",                         "gardiner": "—"},
    "Small_Ring":            {"phonetic": "—",       "meaning": "small ring (ornament)",               "gardiner": "—"},
    "Snake":                 {"phonetic": "ḏȝ",      "meaning": "snake (determinative)",               "gardiner": "I8"},
    "Soldier":               {"phonetic": "—",       "meaning": "soldier, warrior",                    "gardiner": "—"},
    "Star":                  {"phonetic": "sbȝ",     "meaning": "star; door, to teach",                "gardiner": "N14"},
    "Stick":                 {"phonetic": "—",       "meaning": "stick, staff",                        "gardiner": "—"},
    "Swallow":               {"phonetic": "wr",      "meaning": "swallow; great, elder",               "gardiner": "G36"},
    "This":                  {"phonetic": "pn",      "meaning": "this (demonstrative pronoun)",        "gardiner": "—"},
    "To_Be_Dead":            {"phonetic": "mwt",     "meaning": "to die, be dead",                     "gardiner": "—"},
    "To_Protect":            {"phonetic": "sȝ",      "meaning": "to protect, guard",                   "gardiner": "—"},
    "To_Say":                {"phonetic": "ḏd",      "meaning": "to say, speak",                       "gardiner": "—"},
    "Turtle":                {"phonetic": "šȝ",      "meaning": "turtle (evil/enemy det.)",            "gardiner": "I6"},
    "Viper":                 {"phonetic": "f",       "meaning": "horned viper (phonetic: f)",          "gardiner": "I9"},
    "Wall":                  {"phonetic": "inb",     "meaning": "wall",                                "gardiner": "O36"},
    "Water":                 {"phonetic": "n",       "meaning": "water (phonetic: n)",                 "gardiner": "N35"},
    "Woman":                 {"phonetic": "st",      "meaning": "woman (seated determinative)",        "gardiner": "B1"},
    "You":                   {"phonetic": "ṯw",      "meaning": "you (2nd person singular)",           "gardiner": "—"},
}

# Fraction of a box's height used to decide whether two boxes belong to the same row.
ROW_TOLERANCE = 0.5


def _sort_reading_order(boxes, meta):
    """
    Group detections into rows by y-center proximity, then sort each row left-to-right.
    boxes: list of [x1, y1, x2, y2]
    meta:  list of arbitrary objects parallel to boxes
    Returns: list of (box, meta_item) tuples in reading order.
    """
    if not boxes:
        return []

    items = sorted(zip(boxes, meta), key=lambda p: (p[0][1] + p[0][3]) / 2)

    rows = [[items[0]]]
    for item in items[1:]:
        box = item[0]
        y_c = (box[1] + box[3]) / 2
        h   = box[3] - box[1]

        last_box = rows[-1][-1][0]
        last_y_c = (last_box[1] + last_box[3]) / 2
        last_h   = last_box[3] - last_box[1]
        avg_h    = (h + last_h) / 2

        if abs(y_c - last_y_c) < avg_h * ROW_TOLERANCE:
            rows[-1].append(item)
        else:
            rows.append([item])

    ordered = []
    for row in rows:
        row.sort(key=lambda p: (p[0][0] + p[0][2]) / 2)
        ordered.extend(row)
    return ordered


def _preprocess(image_bgr):
    """CLAHE-enhanced grayscale → 3-channel input (same as training preprocessing)."""
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def _draw_annotations(image_rgb, detections):
    """Draw numbered boxes + labels on a copy of image_rgb."""
    output = image_rgb.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 100, 0), 2)
        label = f"{i + 1}. {det['class_name']}  [{det['phonetic']}]"
        cv2.putText(
            output, label,
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 240, 0), 1, cv2.LINE_AA,
        )
    return output


def load_model():
    """Load and return the YOLO model (call once and reuse for the API)."""
    return YOLO(MODEL_PATH)


def run_pipeline(image_path, model=None, conf=0.25, iou=0.4):
    """
    Full detection + reading-order + lookup pipeline.

    Parameters
    ----------
    image_path : str       Path to the input image.
    model      : YOLO      Pre-loaded model (loaded here if None).
    conf       : float     Detection confidence threshold.
    iou        : float     NMS IoU threshold.

    Returns
    -------
    dict with keys:
      annotated_image  – numpy RGB array with numbered bounding boxes
      detections       – list of dicts: box, class_name, confidence,
                         phonetic, meaning, gardiner
      sequence         – ordered list of class-name strings
      transliteration  – space-separated phonetic values (skips "—")
    """
    if model is None:
        model = load_model()

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    model_input = _preprocess(image_bgr)
    results = model(model_input, imgsz=640, conf=conf, iou=iou)[0]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if results.boxes is None or len(results.boxes) == 0:
        return {
            "annotated_image": image_rgb,
            "detections": [],
            "sequence": [],
            "transliteration": "",
        }

    boxes       = results.boxes.xyxy.cpu().numpy().tolist()
    class_ids   = results.boxes.cls.cpu().numpy().astype(int).tolist()
    confidences = results.boxes.conf.cpu().numpy().tolist()
    raw_names   = [model.names[cid] for cid in class_ids]

    ordered = _sort_reading_order(boxes, list(zip(raw_names, confidences)))

    detections = []
    for box, (class_name, conf_val) in ordered:
        info = HIEROGLYPH_LOOKUP.get(class_name, {
            "phonetic": "—",
            "meaning":  class_name,
            "gardiner": "—",
        })
        detections.append({
            "box":        [int(v) for v in box],
            "class_name": class_name,
            "confidence": round(conf_val, 3),
            "phonetic":   info["phonetic"],
            "meaning":    info["meaning"],
            "gardiner":   info["gardiner"],
        })

    sequence        = [d["class_name"] for d in detections]
    transliteration = "  ".join(d["phonetic"] for d in detections if d["phonetic"] != "—")
    annotated       = _draw_annotations(image_rgb, detections)

    return {
        "annotated_image": annotated,
        "detections":      detections,
        "sequence":        sequence,
        "transliteration": transliteration,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path> [conf] [iou]")
        print("  conf  detection confidence threshold (default 0.25)")
        print("  iou   NMS IoU threshold              (default 0.40)")
        sys.exit(1)

    img_path  = sys.argv[1]
    conf_arg  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    iou_arg   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.40

    result = run_pipeline(img_path, conf=conf_arg, iou=iou_arg)

    n = len(result["detections"])
    print(f"\nDetected {n} hieroglyph{'s' if n != 1 else ''} (reading order left-to-right, top-to-bottom):\n")
    print(f"  {'#':<4} {'Class':<25} {'Phonetic':<14} {'Gardiner':<10} Meaning")
    print(f"  {'-'*4} {'-'*25} {'-'*14} {'-'*10} {'-'*30}")
    for i, det in enumerate(result["detections"]):
        print(
            f"  {i+1:<4} {det['class_name']:<25} {det['phonetic']:<14} "
            f"{det['gardiner']:<10} {det['meaning']}"
        )

    if result["transliteration"]:
        print(f"\nTransliteration:  {result['transliteration']}")
    else:
        print("\nNo phonetic values could be determined.")

    out_path = "pipeline_output.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR))
    print(f"\nAnnotated image saved → {out_path}")
