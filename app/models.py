from pydantic import BaseModel
from typing import Optional, List


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


class NormalizedResult(BaseModel):
    raw: RawExtraction
    percentage: Optional[float] = None
    method: Optional[str] = None
    university_matched: Optional[str] = None
    confidence: str = "low"
    classification: Optional[str] = None
    on_10_scale: Optional[float] = None
    on_4_scale: Optional[float] = None
