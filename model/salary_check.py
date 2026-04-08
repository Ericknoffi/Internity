import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re

# Annual USD salary benchmarks by seniority
SALARY_BENCHMARKS = {
    "intern":   (15_000,  60_000),
    "fresher":  (40_000,  80_000),
    "junior":   (55_000, 100_000),
    "mid":      (80_000, 140_000),
    "senior":   (110_000, 220_000),
    "manager":  (120_000, 280_000),
    "director": (150_000, 400_000),
    "default":  (40_000,  150_000),
}

def extract_salary(text: str) -> float | None:
    """
    Extracts annual salary in USD from job text.
    Handles: $50000, $50,000, $50k, $50/hr, 50k/year, 80k-100k
    Returns annual float or None.
    """
    text = text.lower().replace(",", "")

    # Hourly rate  e.g. "$25/hr", "$25 per hour"
    hourly = re.search(r"\$?\s*(\d+\.?\d*)\s*(\/hr|per hour|/hour|hourly)", text)
    if hourly:
        return float(hourly.group(1)) * 40 * 52   # 40hrs/week, 52 weeks

    # Range with k  e.g. "$80k-$100k", "80k to 100k"
    range_k = re.search(r"\$?\s*(\d+\.?\d*)\s*k\s*(?:-|to)\s*\$?\s*(\d+\.?\d*)\s*k", text)
    if range_k:
        low  = float(range_k.group(1)) * 1_000
        high = float(range_k.group(2)) * 1_000
        return (low + high) / 2   # use midpoint

    # Range plain  e.g. "$80000 - $100000"
    range_plain = re.search(r"\$?\s*(\d{4,6})\s*(?:-|to)\s*\$?\s*(\d{4,6})", text)
    if range_plain:
        low  = float(range_plain.group(1))
        high = float(range_plain.group(2))
        return (low + high) / 2

    # k shorthand  e.g. "$80k", "80k/year"
    k_match = re.search(r"\$?\s*(\d+\.?\d*)\s*k", text)
    if k_match:
        return float(k_match.group(1)) * 1_000

    # Plain dollar amount  e.g. "$85000"
    dollar = re.search(r"\$\s*(\d{4,6})\b", text)
    if dollar:
        return float(dollar.group(1))

    # Plain number fallback  e.g. "85000"
    plain = re.search(r"\b(\d{4,6})\b", text)
    if plain:
        return float(plain.group(1))

    return None


def get_seniority(text: str) -> str:
    text_lower = text.lower()
    for key in SALARY_BENCHMARKS:
        if key in text_lower:
            return key
    return "default"


def check_salary_anomaly(text: str) -> dict:
    """
    Returns:
        is_anomaly  (bool)
        salary      (float | None)
        reason      (str)
        risk_score  (0–1 float)
    """
    salary = extract_salary(text)

    if salary is None:
        return {
            "is_anomaly": False,
            "salary": None,
            "reason": "No salary information found in the posting.",
            "risk_score": 0.0
        }

    seniority = get_seniority(text)
    low, high = SALARY_BENCHMARKS[seniority]

    # Unrealistically high — classic scam bait
    if salary > high * 2.5:
        return {
            "is_anomaly": True,
            "salary": salary,
            "reason": f"Salary ${salary:,.0f}/yr is unrealistically high for a {seniority} role (expected up to ${high:,}).",
            "risk_score": 0.8
        }

    # Suspiciously low — possible exploitation
    if salary < low * 0.4:
        return {
            "is_anomaly": True,
            "salary": salary,
            "reason": f"Salary ${salary:,.0f}/yr is suspiciously low for a {seniority} role (expected at least ${low:,}).",
            "risk_score": 0.5
        }

    return {
        "is_anomaly": False,
        "salary": salary,
        "reason": f"Salary ${salary:,.0f}/yr looks normal for a {seniority} role.",
        "risk_score": 0.0
    }