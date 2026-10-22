import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import string
import joblib

# ── Suspicious & High-Risk Lexicon ───────────────────────────────────────────
SUSPICIOUS_PHRASES = [
    "work from home", "no experience needed", "no experience required",
    "guaranteed income", "easy money", "unlimited earning", "be your own boss",
    "no investment required", "weekly payment", "urgent hiring",
    "whatsapp us", "direct joining", "earn daily", "part time earn",
    "apply now limited", "100% job guarantee", "free training",
    "no qualification required", "send cv on whatsapp", "immediate joining",
    "wire transfer", "payment processing", "receive packages", "mystery shopper",
    "cash advance", "bitcoin payment", "crypto investment", "data entry clerk required"
]

URGENCY_KEYWORDS = [
    "urgent", "urgently", "immediate", "immediately", "hurry",
    "act now", "limited time", "don't miss", "instant", "fast cash",
    "apply fast", "positions filling fast", "immediate hiring"
]

PERSONAL_EMAIL_DOMAINS = [
    r"@gmail\.com", r"@yahoo\.com", r"@hotmail\.com",
    r"@outlook\.com", r"@aol\.com", r"@icloud\.com", r"@mail\.com"
]

# ── Preprocessing & Vectorization ────────────────────────────────────────────
_vectorizer = None

def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "saved_model",
            "tfidf_vectorizer.pkl"
        )
        if os.path.exists(model_path):
            _vectorizer = joblib.load(model_path)
    return _vectorizer

def clean_text(text: str) -> str:
    """
    Normalized text cleaner for TF-IDF feature extraction.
    Removes raw URLs, normalizes whitespace, and standardizes casing.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)        # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)           # sanitize symbols
    text = re.sub(r"\s+", " ", text).strip()            # collapse whitespace
    return text

def extract_suspicious_flags(text: str) -> list:
    """Return list of suspicious scam phrases found in raw text."""
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    return [phrase for phrase in SUSPICIOUS_PHRASES if phrase in text_lower]

def extract_stylometrics(text: str) -> dict:
    """
    Extract 10 linguistic stylometric features from the raw input text.
    Preserves casing, punctuation, and structural nuances lost in regex stripping.
    """
    if not text or not isinstance(text, str):
        return {
            "uppercase_ratio": 0.0,
            "exclamation_density": 0.0,
            "question_density": 0.0,
            "currency_density": 0.0,
            "digit_density": 0.0,
            "urgency_keyword_count": 0,
            "whatsapp_telegram_trigger": 0,
            "personal_email_count": 0,
            "scam_keyword_density": 0.0,
            "excessive_punctuation_ratio": 0.0,
            "stylometric_risk_score": 0.0
        }

    raw_len = max(len(text), 1)
    len_per_100 = max(len(text) / 100.0, 1.0)
    words = text.split()
    total_words = max(len(words), 1)
    text_lower = text.lower()

    # 1. Uppercase character ratio
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = round(uppercase_count / raw_len, 4)

    # 2. Exclamation mark density (per 100 characters)
    exclamation_count = text.count("!")
    exclamation_density = round(exclamation_count / len_per_100, 4)

    # 3. Question mark density (per 100 characters)
    question_count = text.count("?")
    question_density = round(question_count / len_per_100, 4)

    # 4. Currency symbol density ($, €, £, ₹, ¥) per 100 characters
    currency_count = len(re.findall(r"[\$\€\£\₹\¥]", text))
    currency_density = round(currency_count / len_per_100, 4)

    # 5. Digit character density
    digit_count = sum(1 for c in text if c.isdigit())
    digit_density = round(digit_count / raw_len, 4)

    # 6. Urgency keyword count
    urgency_keyword_count = sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)

    # 7. WhatsApp / Telegram communication triggers
    whatsapp_telegram_trigger = 1 if re.search(r"(?i)\b(whatsapp|telegram|wa\.me|t\.me)\b", text) else 0

    # 8. Personal free email domain detection (unprofessional contact)
    personal_email_count = sum(len(re.findall(dom, text_lower)) for dom in PERSONAL_EMAIL_DOMAINS)

    # 9. Scam keyword density (suspicious phrases relative to word count)
    suspicious_found = [p for p in SUSPICIOUS_PHRASES if p in text_lower]
    scam_keyword_density = round(len(suspicious_found) / total_words, 4)

    # 10. Excessive / repeated punctuation (e.g., !!, ??, $$, ...)
    repeated_punct_count = len(re.findall(r"[!?\$\*]{2,}", text))
    excessive_punctuation_ratio = round(repeated_punct_count / len_per_100, 4)

    # Composite stylometric risk heuristic (bounded 0.0 - 1.0)
    risk_score = min(
        (uppercase_ratio * 1.5) +
        (exclamation_density * 0.1) +
        (currency_density * 0.15) +
        (urgency_keyword_count * 0.1) +
        (whatsapp_telegram_trigger * 0.35) +
        (personal_email_count * 0.25) +
        (len(suspicious_found) * 0.15),
        1.0
    )

    return {
        "uppercase_ratio": uppercase_ratio,
        "exclamation_density": exclamation_density,
        "question_density": question_density,
        "currency_density": currency_density,
        "digit_density": digit_density,
        "urgency_keyword_count": urgency_keyword_count,
        "whatsapp_telegram_trigger": whatsapp_telegram_trigger,
        "personal_email_count": personal_email_count,
        "scam_keyword_density": scam_keyword_density,
        "excessive_punctuation_ratio": excessive_punctuation_ratio,
        "stylometric_risk_score": round(risk_score, 4)
    }

def vectorize(text: str):
    """Transform input text into TF-IDF sparse matrix feature vector."""
    vec = get_vectorizer()
    if vec is None:
        raise RuntimeError("TF-IDF vectorizer artifact not found in saved_model/")
    cleaned = clean_text(text)
    return vec.transform([cleaned])