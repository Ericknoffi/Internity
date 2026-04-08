import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import string
import joblib

# ── Load saved vectorizer ─────────────────────────────────────────────────────
vectorizer = joblib.load("saved_model/tfidf_vectorizer.pkl")

SUSPICIOUS_PHRASES = [
    "work from home", "no experience needed", "guaranteed income",
    "easy money", "unlimited earning", "be your own boss",
    "no investment required", "weekly payment", "urgent hiring",
    "whatsapp us", "direct joining", "earn daily", "part time earn",
    "apply now limited", "100% job guarantee", "free training",
    "no qualification required", "send cv on whatsapp"
]

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()           # collapse whitespace
    return text

def extract_suspicious_flags(text: str) -> list:
    """Return list of suspicious phrases found in raw text."""
    text_lower = text.lower()
    return [phrase for phrase in SUSPICIOUS_PHRASES if phrase in text_lower]

def vectorize(text: str):
    cleaned = clean_text(text)
    return vectorizer.transform([cleaned])