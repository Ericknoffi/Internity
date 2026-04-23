import re

# Keywords that strongly indicate a job posting
JOB_KEYWORDS = [
    "job", "position", "role", "hiring", "vacancy", "opening",
    "apply", "application", "candidate", "applicant", "recruit",
    "resume", "cv", "interview", "salary", "compensation", "benefits",
    "experience", "qualification", "requirement", "skill", "degree",
    "responsibilities", "duties", "company", "employer", "organization",
    "full time", "part time", "remote", "work from", "office",
    "internship", "fresher", "engineer", "developer", "manager",
    "analyst", "executive", "officer", "assistant", "associate",
    "employment", "join", "team", "department", "working hours"
]

def is_job_posting(text: str) -> dict:
    """
    Checks if submitted text looks like a job posting.
    Returns:
        is_valid  (bool)
        score     (float) - how confident we are it is a job posting
        reason    (str)   - explanation if rejected
    """
    if not text or len(text.strip()) < 50:
        return {
            "is_valid": False,
            "score":    0.0,
            "reason":   "Text is too short. Please paste a complete job description (at least 50 characters)."
        }

    text_lower = text.lower()
    words      = re.findall(r"\b\w+\b", text_lower)
    total      = len(words)

    if total < 20:
        return {
            "is_valid": False,
            "score":    0.0,
            "reason":   "Too few words. Please paste the full job description, not just a title or snippet."
        }

    # Count how many job keywords appear
    hits = [kw for kw in JOB_KEYWORDS if kw in text_lower]
    score = min(len(hits) / 6.0, 1.0)   # 6+ hits = confident it's a job post

    if score < 0.25:
        return {
            "is_valid": False,
            "score":    score,
            "reason": (
                "This doesn't look like a job posting. "
                "Please paste a job description that includes details like "
                "role, responsibilities, requirements, salary, or company information."
            )
        }

    return {
        "is_valid": True,
        "score":    score,
        "reason":   ""
    }
