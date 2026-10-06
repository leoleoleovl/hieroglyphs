import base64
import tempfile
import os

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import load_model, run_pipeline

app = FastAPI(title="Hieroglyph Translator")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Allow requests from the Expo mobile app (any origin is fine for local use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup — not per request
_model = None

@app.on_event("startup")
def startup():
    global _model
    _model = load_model()
    print("Model loaded and ready.")


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = 0.10,
    iou: float = 0.40,
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Write upload to a temp file so OpenCV can read it
    contents = await file.read()
    suffix = os.path.splitext(file.filename or ".jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = run_pipeline(tmp_path, model=_model, conf=conf, iou=iou)
    finally:
        os.unlink(tmp_path)

    # Encode annotated image as base64 JPEG so the mobile app can display it
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR))
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "count":           len(result["detections"]),
        "sequence":        result["sequence"],
        "transliteration": result["transliteration"],
        "detections":      result["detections"],
        "annotated_image": img_b64,
    }
