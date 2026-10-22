import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import html
import joblib
import numpy as np
from utils.preprocessor import clean_text, vectorize

# ── Model & Vectorizer Loader ────────────────────────────────────────────────
_lr_model = None
_vectorizer = None

def get_artifacts():
    global _lr_model, _vectorizer
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _lr_model is None:
        lr_path = os.path.join(base_dir, "saved_model", "lr_model.pkl")
        if os.path.exists(lr_path):
            _lr_model = joblib.load(lr_path)
    if _vectorizer is None:
        vec_path = os.path.join(base_dir, "saved_model", "tfidf_vectorizer.pkl")
        if os.path.exists(vec_path):
            _vectorizer = joblib.load(vec_path)
    return _lr_model, _vectorizer

def build_highlight_html(text: str, fake_set: dict, real_set: dict) -> str:
    """
    Constructs safe, XSS-sanitized HTML highlighting words according to linear attribution.
    Red highlight   = word pushed prediction toward FAKE
    Green highlight = word pushed prediction toward REAL
    """
    words = text.split()
    tokens = []

    for word in words:
        # Strip trailing/leading punctuation for dictionary lookup
        lookup = word.lower().strip(".,!?;:\"'()[]{}")
        safe_word = html.escape(word)

        if lookup in fake_set:
            intensity = fake_set[lookup]
            alpha = round(0.2 + 0.6 * intensity, 2)   # Scale 0.20 to 0.80
            span = (
                f'<span class="shap-fake" '
                f'style="background:rgba(226,75,74,{alpha});'
                f'border-radius:3px;padding:1px 3px;" '
                f'title="Pushes toward FAKE (impact: +{intensity:.2f})">'
                f'{safe_word}</span>'
            )
            tokens.append(span)

        elif lookup in real_set:
            intensity = real_set[lookup]
            alpha = round(0.2 + 0.6 * intensity, 2)
            span = (
                f'<span class="shap-real" '
                f'style="background:rgba(99,153,34,{alpha});'
                f'border-radius:3px;padding:1px 3px;" '
                f'title="Pushes toward REAL (impact: -{intensity:.2f})">'
                f'{safe_word}</span>'
            )
            tokens.append(span)

        else:
            tokens.append(safe_word)

    return " ".join(tokens)


def get_top_shap_words(text: str, top_n: int = 15) -> dict:
    """
    Fast analytical linear feature attribution:
        Attribution(w) = beta_w * tfidf_w
    
    Replaces slow/unstable C++ SHAP dependency with <1 ms closed-form attribution
    for linear models with TF-IDF features.
    
    Returns:
        fake_words  — list of (word, score) pushing toward fake (positive)
        real_words  — list of (word, score) pushing toward real (negative)
        html        — sanitized, highlighted HTML string
    """
    lr_model, vectorizer = get_artifacts()
    if lr_model is None or vectorizer is None:
        return {"fake_words": [], "real_words": [], "html": html.escape(text)}

    cleaned = clean_text(text)
    if not cleaned:
        return {"fake_words": [], "real_words": [], "html": html.escape(text)}

    X_vec = vectorizer.transform([cleaned])
    cx = X_vec.tocoo()

    coefs = lr_model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()

    word_attributions = []
    for col, val in zip(cx.col, cx.data):
        attr = float(coefs[col] * val)
        word_attributions.append((feature_names[col], attr))

    # Sort descending by absolute attribution magnitude
    word_attributions.sort(key=lambda x: abs(x[1]), reverse=True)

    fake_words = [(w, v) for w, v in word_attributions if v > 0][:top_n]
    real_words = [(w, v) for w, v in word_attributions if v < 0][:top_n]

    max_fake = max((v for _, v in fake_words), default=1.0)
    max_real = max((abs(v) for _, v in real_words), default=1.0)
    if max_fake == 0:
        max_fake = 1.0
    if max_real == 0:
        max_real = 1.0

    fake_set = {w: v / max_fake for w, v in fake_words}
    real_set = {w: abs(v) / max_real for w, v in real_words}

    highlighted_html = build_highlight_html(cleaned, fake_set, real_set)

    return {
        "fake_words": fake_words,
        "real_words": real_words,
        "html": highlighted_html
    }