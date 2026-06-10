from pydantic import BaseModel, field_validator, Field
from typing import Optional, List, Any


class RawExtraction(BaseModel):
    institution:          Optional[str]   = Field(None, description="University or board name as printed on document")
    student_name:         Optional[str]   = Field(None, description="Student full name")
    degree:               Optional[str]   = Field(None, description="Degree e.g. B.E., B.Tech, B.Sc, M.Tech")
    specialization:       Optional[str]   = Field(None, description="Branch / specialization")
    year_of_passing:      Optional[str]   = Field(None, description="Year of passing or exam year")
    marks_scored:         Optional[float] = Field(None, description="Grand total marks obtained (from summary row)")
    total_marks:          Optional[float] = Field(None, description="Grand total maximum marks")
    percentage:           Optional[float] = Field(None, description="Percentage if explicitly printed on document")
    grade:                Optional[str]   = Field(None, description="Overall letter grade e.g. O, A+, A, B+")
    gpa:                  Optional[float] = Field(None, description="GPA value if shown")
    cgpa:                 Optional[float] = Field(None, description="CGPA value if shown")
    grade_scale:          Optional[str]   = Field(None, description="Scale e.g. 10, 7, 4.0")
    result:               Optional[str]   = Field(None, description="Result class as printed e.g. FIRST CLASS WITH DISTINCTION")
    cgpa_formula_on_cert: Optional[str]   = Field(None, description="CGPA-to-percentage formula if printed on document")

    @field_validator("year_of_passing", "institution", "student_name", "degree",
                     "specialization", "grade", "grade_scale", "result",
                     "cgpa_formula_on_cert", mode="before")
    @classmethod
    def coerce_to_str(cls, v: Any) -> Any:
        if v is not None and not isinstance(v, str):
            return str(v)
        return v


class NormalizedResult(BaseModel):
    raw:                RawExtraction  = Field(...,  description="Raw values extracted directly from the document by AI")
    llm_json:           Optional[Any]  = Field(None, description="Exact JSON returned by the vision model before any processing")
    percentage:         Optional[float]= Field(None, description="Normalized percentage. Null if could not be determined from document")
    method:             Optional[str]  = Field(None, description="How percentage was derived e.g. 'marks 850/1200' or 'cgpa * 10 (anna_university)'")
    university_matched: Optional[str]  = Field(None, description="Matched university rule key. 'fallback_ugc' means unknown university — UGC standard formula used")
    confidence:         str            = Field("low", description="University match confidence: high=exact match, medium=fuzzy, low=unknown")
    classification:     Optional[str]  = Field(None, description="Academic classification: First Class with Distinction / First Class / Second Class / Pass")
    on_10_scale:        Optional[float]= Field(None, description="Percentage expressed on a 10-point scale (percentage / 10)")
    on_4_scale:         Optional[float]= Field(None, description="Percentage expressed on a 4-point GPA scale (percentage / 100 * 4)")
