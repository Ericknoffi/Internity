import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
from utils.preprocessor import vectorize, extract_suspicious_flags
from model.salary_check import check_salary_anomaly

# ── Load models once at import time ──────────────────────────────────────────
lr_model = joblib.load("saved_model/lr_model.pkl")
nb_model = joblib.load("saved_model/nb_model.pkl")

def get_risk_label(score: float) -> str:
    if score >= 0.65:
        return "High Risk"
    elif score >= 0.35:
        return "Suspicious"
    else:
        return "Safe"

def predict(text: str) -> dict:
    """
    Run full prediction pipeline on raw job text.
    Returns a structured result dict for the Flask route.
    """

    # 1. Vectorize
    X = vectorize(text)

    # 2. ML predictions
    lr_prob  = lr_model.predict_proba(X)[0][1]   # prob of fake
    nb_prob  = nb_model.predict_proba(X)[0][1]

    # Ensemble: weighted average (LR is more reliable)
    ml_score = 0.65 * lr_prob + 0.35 * nb_prob

    # 3. Rule-based: suspicious phrases
    flags = extract_suspicious_flags(text)
    phrase_score = min(len(flags) * 0.08, 0.4)   # cap at 0.4

    # 4. Salary anomaly
    salary_result = check_salary_anomaly(text)
    salary_score  = salary_result["risk_score"]

    # 5. Final combined score
    final_score = min(
        ml_score * 0.60 +
        phrase_score * 0.25 +
        salary_score * 0.15,
        1.0
    )

    return {
        "final_score":    round(final_score, 4),
        "risk_label":     get_risk_label(final_score),
        "ml_score":       round(ml_score, 4),
        "lr_prob":        round(lr_prob, 4),
        "nb_prob":        round(nb_prob, 4),
        "phrase_score":   round(phrase_score, 4),
        "suspicious_phrases": flags,
        "salary":         salary_result,
    }