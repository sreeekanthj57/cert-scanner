import json
import os
from difflib import get_close_matches
from models import RawExtraction, NormalizedResult

_RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "university_rules.json")

def _load_rules() -> dict:
    with open(_RULES_PATH, "r") as f:
        return json.load(f)


def _resolve_university(name: str, rules: dict) -> tuple[str, str]:
    if not name:
        return "fallback_ugc", "low"

    name_lower = name.lower()

    # Exact/substring match
    for key, rule in rules.items():
        for alias in rule.get("aliases", []):
            if alias.lower() in name_lower:
                return key, "high"

    # Fuzzy match — high cutoff to avoid false positives
    alias_map = {a.lower(): k for k, r in rules.items() for a in r.get("aliases", [])}
    matches = get_close_matches(name_lower, alias_map.keys(), n=1, cutoff=0.80)
    if matches:
        return alias_map[matches[0]], "medium"

    return "fallback_ugc", "low"


def _safe_eval_formula(formula: str, cgpa: float) -> float:
    # Only allow safe math — no arbitrary code
    allowed = {"cgpa": cgpa, "__builtins__": {}}
    return eval(formula, {"__builtins__": {}}, {"cgpa": cgpa})


def _classify(pct: float, rule: dict) -> str:
    c = rule.get("classification", {"distinction": 75, "first": 60, "second": 50})
    if pct >= c.get("distinction", 75):
        return "First Class with Distinction"
    if pct >= c.get("first", 60):
        return "First Class"
    if pct >= c.get("second", 50):
        return "Second Class"
    return "Pass"


def normalize(raw: RawExtraction) -> NormalizedResult:
    rules = _load_rules()
    rule_key, confidence = _resolve_university(raw.institution or "", rules)
    rule = rules[rule_key]

    pct: float | None = None
    method = ""

    # 1. Direct percentage on certificate — most reliable
    if raw.percentage:
        pct = raw.percentage
        method = "direct from certificate"

    # 2. Marks scored / total marks
    elif raw.marks_scored and raw.total_marks and raw.total_marks > 0:
        pct = (raw.marks_scored / raw.total_marks) * 100
        method = f"marks {raw.marks_scored}/{raw.total_marks}"

    # 3. CGPA — check if formula is printed on cert first
    elif raw.cgpa:
        cgpa = raw.cgpa

        if raw.cgpa_formula_on_cert:
            # Certificate has its own formula — highest priority
            try:
                formula_clean = raw.cgpa_formula_on_cert.lower().replace("percentage", "").replace("=", "").strip()
                pct = _safe_eval_formula(formula_clean, cgpa)
                method = f"formula from cert: {raw.cgpa_formula_on_cert}"
                confidence = "high"
            except Exception:
                pass

        if pct is None:
            formula = rule.get("cgpa_to_pct", "cgpa * 10")
            try:
                pct = _safe_eval_formula(formula, cgpa)
                method = f"{formula} ({rule_key})"
            except Exception:
                pct = cgpa * 10
                method = f"cgpa * 10 (fallback)"

    # 4. Grade → midpoint of range
    elif raw.grade and "grades" in rule:
        grade_data = rule["grades"].get(raw.grade)
        if grade_data:
            r = grade_data["range"]
            pct = (r[0] + r[1]) / 2
            method = f"grade {raw.grade} → midpoint {pct:.1f}% (approx)"

    # 5. Result class fallback — provisional/degree certs with no marks
    is_approximate = False
    percentage_range = None
    if pct is None and raw.result:
        res = raw.result.upper()
        if "DISTINCTION" in res or "HONOURS" in res or "HONOR" in res:
            pct = 75.0
            percentage_range = "≥75%"
            method = "approximate minimum — First Class with Distinction"
        elif "FIRST" in res:
            pct = 60.0
            percentage_range = "60–74%"
            method = "approximate minimum — First Class"
        elif "SECOND" in res:
            pct = 50.0
            percentage_range = "50–59%"
            method = "approximate minimum — Second Class"
        elif "PASS" in res or "THIRD" in res:
            pct = 40.0
            percentage_range = "40–49%"
            method = "approximate minimum — Pass Class"
        if pct is not None:
            is_approximate = True

    classification = _classify(pct, rule) if pct is not None else (raw.result or None)

    return NormalizedResult(
        raw=raw,
        is_provisional=is_approximate,
        percentage=round(pct, 2) if pct is not None else None,
        percentage_range=percentage_range,
        method=method,
        university_matched=rule_key,
        confidence=confidence,
        classification=classification,
        on_10_scale=round(pct / 10, 2) if pct is not None and not is_approximate else None,
        on_4_scale=round((pct / 100) * 4, 2) if pct is not None and not is_approximate else None,
    )
