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

Extracts and normalizes academic scores from Indian degree certificates, marksheets, and provisional certificates using AI vision (Gemini via OpenRouter).

---

### Supported Documents
| Type | What is extracted |
|---|---|
| **Marksheet** | Grand total marks, percentage, CGPA, grade |
| **Degree Certificate** | Result class, CGPA if present |
| **Provisional Certificate** | Result class, CGPA if present |
| **Transcript** | Semester marks, CGPA |

---

### How Percentage is Computed (priority order)
1. **Direct** — percentage explicitly printed on the document
2. **Marks** — `(marks_scored / total_marks) × 100`
3. **CGPA** — using formula printed on cert, or university-specific formula, or `CGPA × 10` (UGC standard fallback)
4. **Grade letter** — midpoint of grade range for that university

If none of the above are available (result class only, or poor scan), `percentage` will be `null`.

---

### Response Fields

| Field | Type | Description |
|---|---|---|
| `raw` | object | Exactly what the AI extracted from the document |
| `llm_json` | object | Raw JSON returned by the vision model |
| `percentage` | float \\| null | Normalized percentage — null if could not be determined |
| `classification` | string \\| null | First Class with Distinction / First Class / Second Class / Pass |
| `on_10_scale` | float \\| null | Equivalent on 10-point scale |
| `on_4_scale` | float \\| null | Equivalent on 4-point GPA scale |
| `method` | string | How percentage was derived e.g. `"marks 850/1200"` |
| `university_matched` | string | Matched university rule key or `fallback_ugc` |
| `confidence` | string | `high` / `medium` / `low` — how well university was identified |

---

### Quality Check (frontend)
- **Green** — `percentage`, `marks`, or `cgpa` successfully extracted
- **Red** — none of the above found; manual verification needed

---

### Error Codes
| Code | Meaning |
|---|---|
| 400 | Bad file type, file too large, or invalid URL |
| 422 | Image could not be decoded/read |
| 502 | AI model call failed (check `OPENROUTER_API_KEY`) |
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
