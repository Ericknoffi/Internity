import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
from utils.preprocessor import vectorize, extract_suspicious_flags, extract_stylometrics
from model.salary_check import check_salary_anomaly

# ── Load models once at import time ──────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lr_model = joblib.load(os.path.join(_BASE_DIR, "saved_model", "lr_model.pkl"))
# nb_model.pkl contains the calibrated SGDClassifier (Modified Huber Loss)
nb_model = joblib.load(os.path.join(_BASE_DIR, "saved_model", "nb_model.pkl"))

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
    Combines:
      1. Calibrated Dual-Model Ensemble (65% Cost-Sensitive LR + 35% Calibrated SGD)
      2. 10-Feature Linguistic Stylometrics & Suspicious Phrase Detection
      3. Salary Anomaly Detection
    Returns a structured result dict for the Flask route and UI templates.
    """
    # 1. Vectorize text with 15k-feature Sublinear Multi-Gram TF-IDF
    X = vectorize(text)

    # 2. ML predictions: 65% Primary LR + 35% Secondary Calibrated SGD
    lr_prob = float(lr_model.predict_proba(X)[0][1])   # probability of fake
    nb_prob = float(nb_model.predict_proba(X)[0][1])   # calibrated SGD prob
    ml_score = 0.65 * lr_prob + 0.35 * nb_prob

    # 3. Linguistic Stylometrics & Rule-based flags
    stylometrics = extract_stylometrics(text)
    flags = extract_suspicious_flags(text)
    
    # Phrase score incorporates detected scam phrases and stylometric indicators
    base_phrase_score = min(len(flags) * 0.08, 0.4)
    stylo_contribution = stylometrics["stylometric_risk_score"] * 0.2
    phrase_score = min(base_phrase_score + stylo_contribution, 0.5)

    # 4. Salary anomaly check
    salary_result = check_salary_anomaly(text)
    salary_score = salary_result.get("risk_score", 0.0)

    # 5. Final combined weighted risk score
    final_score = min(
        (ml_score * 0.60) +
        (phrase_score * 0.25) +
        (salary_score * 0.15),
        1.0
    )

    return {
        "final_score": round(final_score, 4),
        "risk_label": get_risk_label(final_score),
        "ml_score": round(ml_score, 4),
        "lr_prob": round(lr_prob, 4),
        "nb_prob": round(nb_prob, 4),
        "sgd_prob": round(nb_prob, 4),
        "phrase_score": round(phrase_score, 4),
        "suspicious_phrases": flags,
        "salary": salary_result,
        "stylometrics": stylometrics,
    }