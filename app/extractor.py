import os
import base64
import json
import re
from openai import OpenAI
from models import RawExtraction

PROMPT = """
You are analyzing an Indian academic marksheet, degree certificate, or transcript.

Extract ONLY what is explicitly visible. Do not guess or compute values yourself.

IMPORTANT — Indian marksheets always have a GRAND TOTAL / SUMMARY row at the bottom of the marks table showing total marks obtained vs maximum marks (e.g. "Total: 850 / 1200" or "Grand Total: 720 out of 900"). Find this row and use it for marks_scored and total_marks. Also look for a printed percentage, CGPA, or GPA near that summary row.

Return a single valid JSON object — no markdown, no explanation, no code fences:
{
  "institution": "full university/board name as printed",
  "student_name": "student full name",
  "degree": "degree name e.g. B.E., B.Tech, B.Sc, M.Tech, MBA",
  "specialization": "branch/specialization e.g. Computer Science",
  "year_of_passing": "year of passing or exam year",
  "marks_scored": <grand total marks obtained — number or null>,
  "total_marks": <grand total maximum marks — number or null>,
  "percentage": <percentage if explicitly printed on document — number or null>,
  "grade": "overall letter grade if shown e.g. O, A+, A, B+",
  "gpa": <GPA value if shown — number or null>,
  "cgpa": <CGPA value if shown — number or null>,
  "grade_scale": "scale e.g. 10, 7, 4.0 — null if not shown",
  "result": "result class exactly as printed e.g. FIRST CLASS WITH DISTINCTION, FIRST CLASS, PASS",
  "cgpa_formula_on_cert": "if a CGPA-to-percentage conversion formula is printed anywhere on the document copy it exactly, else null",
}

Rules:
- marks_scored / total_marks: use the GRAND TOTAL row only, not individual subject rows
- percentage: only if explicitly printed — do NOT compute it
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
