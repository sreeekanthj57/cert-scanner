import os
import base64
import json
import re
from openai import OpenAI
from models import RawExtraction

PROMPT = """
You are analyzing an Indian academic certificate, marksheet, or degree document.

Extract ONLY what is explicitly visible. Do not guess or infer values not shown.

Return a single valid JSON object — no markdown, no explanation, no code fences:
{
  "institution": "full institution name as printed",
  "student_name": "student name",
  "degree": "degree name e.g. B.E., B.Tech, B.Sc, M.Tech",
  "specialization": "branch/specialization e.g. Computer Science",
  "year_of_passing": "year",
  "marks_scored": <number or null>,
  "total_marks": <number or null>,
  "percentage": <number or null>,
  "grade": "letter grade if shown e.g. O, A+, A, B+",
  "gpa": <number or null>,
  "cgpa": <number or null>,
  "grade_scale": "scale shown on cert e.g. 10, 7, 4.0",
  "result": "result class if shown e.g. FIRST CLASS WITH DISTINCTION",
  "cgpa_formula_on_cert": "if a conversion formula is printed on the certificate, copy it exactly, else null",
  "subject_wise": [
    { "subject": "name", "marks": <number>, "total": <number>, "grade": "letter or null" }
  ]
}

Rules:
- percentage: use value directly if shown; do NOT compute it yourself
- cgpa/gpa: include the scale if visible (look for /10 or /4 notation)
- subject_wise: include all rows from the marks table; omit if no table visible
- Return null for any field not found
"""


def extract_from_image(image_bytes: bytes) -> RawExtraction:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    response = client.chat.completions.create(
        model=os.environ.get("VISION_MODEL", "google/gemini-2.0-flash-lite"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        max_tokens=2000,
    )

    raw_text = response.choices[0].message.content.strip()
    # Strip markdown fences if model adds them
    raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()

    data = json.loads(raw_text)
    return RawExtraction(**data)
