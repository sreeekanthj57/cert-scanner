import os
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from preprocess import preprocess_certificate
from extractor import extract_from_image
from normalizer import normalize
from models import NormalizedResult

app = FastAPI(
    title="Certificate Scanner API",
    description="""
## Indian Academic Certificate Scanner

Extracts and normalizes academic scores from Indian degree certificates, marksheets, and provisional certificates using AI vision.

### Supported Documents
- **Marksheets** — extracts grand total marks, computes percentage
- **Provisional Certificates** — extracts result class, shows approximate range
- **Degree Certificates** — extracts result class and CGPA if present
- **Transcripts** — semester-wise academic records

### Possible Outcomes

| Field | Description |
|---|---|
| `percentage` | Exact if found on cert, or computed from marks/CGPA |
| `percentage_range` | e.g. `"60–74%"` — shown when only result class is available |
| `is_provisional` | `true` when values are approximate (provisional/degree cert) |
| `classification` | First Class with Distinction / First Class / Second Class / Pass |
| `on_10_scale` | Percentage converted to 10-point scale |
| `on_4_scale` | Percentage converted to 4-point GPA scale |
| `method` | How the percentage was derived |
| `confidence` | `high` / `medium` / `low` — university match quality |

### Quality Indicators
- **Green** — exact data extracted from document
- **Orange** — approximate values derived from result class only
- **Red** — extraction failed, manual verification needed
""",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
MAX_SIZE_MB = 10
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")


def _process(raw_bytes: bytes) -> NormalizedResult:
    try:
        preprocess_certificate(raw_bytes)
    except Exception as e:
        raise HTTPException(422, f"Invalid or unreadable image: {e}")
    try:
        extraction, llm_json = extract_from_image(raw_bytes)
    except Exception as e:
        raise HTTPException(502, f"AI extraction failed: {e}")
    result = normalize(extraction)
    result.llm_json = llm_json
    return result


@app.get("/health", tags=["System"])
def health():
    """Check if the service is running."""
    return {"status": "ok"}


@app.post(
    "/api/scan",
    response_model=NormalizedResult,
    tags=["Scan"],
    summary="Scan certificate from file upload",
    description="Upload a certificate image (JPEG/PNG/WEBP/TIFF, max 10MB). Returns extracted and normalized academic scores.",
)
async def scan_certificate(file: UploadFile = File(..., description="Certificate image file")):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported type: {file.content_type}. Use JPEG, PNG, WEBP or TIFF.")
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max {MAX_SIZE_MB}MB.")
    return _process(raw_bytes)


class ScanUrlRequest(BaseModel):
    url: str

    model_config = {
        "json_schema_extra": {
            "example": {"url": "https://example.com/marksheet.jpg"}
        }
    }


@app.post(
    "/api/scan-url",
    response_model=NormalizedResult,
    tags=["Scan"],
    summary="Scan certificate from image URL",
    description="Provide a publicly accessible URL to a certificate image. The server fetches and processes it.",
)
async def scan_from_url(req: ScanUrlRequest):
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(req.url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(400, f"Could not fetch URL: HTTP {e.response.status_code}")
    except Exception as e:
        raise HTTPException(400, f"Could not fetch URL: {e}")

    content_type = resp.headers.get("content-type", "")
    if not any(t in content_type for t in ["image/jpeg", "image/png", "image/webp", "image/tiff", "application/octet-stream"]):
        raise HTTPException(400, f"URL does not point to a supported image. Got: {content_type}")

    raw_bytes = resp.content
    if len(raw_bytes) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"Image too large. Max {MAX_SIZE_MB}MB.")

    return _process(raw_bytes)


# Serve frontend — must be after all API routes so /docs and /redoc still work
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
