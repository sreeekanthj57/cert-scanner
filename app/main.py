import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from preprocess import preprocess_certificate
from extractor import extract_from_image
from normalizer import normalize
from models import NormalizedResult

app = FastAPI(title="Certificate Scanner", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
MAX_SIZE_MB = 10


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/scan", response_model=NormalizedResult)
async def scan_certificate(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use JPEG, PNG, WEBP or TIFF.")

    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max {MAX_SIZE_MB}MB.")

    try:
        preprocess_certificate(raw_bytes)  # validate image is readable
    except Exception as e:
        raise HTTPException(422, f"Image preprocessing failed: {e}")

    try:
        # Send original image — vision LLMs read color better than B&W threshold
        extraction, llm_json = extract_from_image(raw_bytes)
    except Exception as e:
        raise HTTPException(502, f"AI extraction failed: {e}")

    result = normalize(extraction)
    result.llm_json = llm_json
    return result


# Serve frontend at root — must be last
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
