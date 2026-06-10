from pydantic import BaseModel, field_validator
from typing import Optional, List, Any


class SubjectScore(BaseModel):
    subject: str
    marks: Optional[float] = None
    total: Optional[float] = None
    grade: Optional[str] = None


class RawExtraction(BaseModel):
    institution: Optional[str] = None
    student_name: Optional[str] = None
    degree: Optional[str] = None
    specialization: Optional[str] = None
    year_of_passing: Optional[str] = None
    marks_scored: Optional[float] = None
    total_marks: Optional[float] = None
    percentage: Optional[float] = None
    grade: Optional[str] = None
    gpa: Optional[float] = None
    cgpa: Optional[float] = None
    grade_scale: Optional[str] = None
    result: Optional[str] = None
    cgpa_formula_on_cert: Optional[str] = None

    @field_validator("year_of_passing", "institution", "student_name", "degree",
                     "specialization", "grade", "grade_scale", "result",
                     "cgpa_formula_on_cert", mode="before")
    @classmethod
    def coerce_to_str(cls, v: Any) -> Any:
        if v is not None and not isinstance(v, str):
            return str(v)
        return v


class NormalizedResult(BaseModel):
    raw: RawExtraction
    llm_json: Optional[Any] = None
    is_provisional: bool = False
    percentage: Optional[float] = None
    percentage_range: Optional[str] = None   # e.g. "60–74%" when exact not available
    method: Optional[str] = None
    university_matched: Optional[str] = None
    confidence: str = "low"
    classification: Optional[str] = None
    on_10_scale: Optional[float] = None
    on_4_scale: Optional[float] = None
