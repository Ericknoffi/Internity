import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shap
import joblib
import numpy as np
from utils.preprocessor import clean_text, vectorize

# ── Load model and vectorizer once ───────────────────────────────────────────
lr_model   = joblib.load("saved_model/lr_model.pkl")
vectorizer = joblib.load("saved_model/tfidf_vectorizer.pkl")

# ── Build SHAP explainer ──────────────────────────────────────────────────────
# LinearExplainer is the right choice for Logistic Regression + TF-IDF
# masker=shap.maskers.Independent handles sparse TF-IDF matrices correctly
explainer = shap.LinearExplainer(
    lr_model,
    masker=shap.maskers.Independent(
        np.zeros((1, len(vectorizer.get_feature_names_out()))),
        max_samples=100
    )
)

def get_top_shap_words(text: str, top_n: int = 15) -> dict:
    """
    Returns the top words that pushed the prediction toward FAKE,
    and top words that pushed it toward REAL.

    Returns:
        fake_words  — list of (word, shap_value) pushing toward fake
        real_words  — list of (word, shap_value) pushing toward real
        html        — ready-to-render highlighted HTML string
    """
    cleaned   = clean_text(text)
    X_vec     = vectorize(text)                          # sparse matrix
    X_dense   = X_vec.toarray()                          # SHAP needs dense

    shap_values = explainer.shap_values(X_dense)[0]     # 1D array of shape (n_features,)

    feature_names = vectorizer.get_feature_names_out()

    # Map feature index → shap value, filter only non-zero (words present in text)
    present_indices = X_dense[0].nonzero()[0]
    word_shap = [
        (feature_names[i], shap_values[i])
        for i in present_indices
    ]

    # Sort by absolute shap value
    word_shap.sort(key=lambda x: abs(x[1]), reverse=True)

    # Split into fake-pushing (positive shap) and real-pushing (negative shap)
    fake_words = [(w, v) for w, v in word_shap if v > 0][:top_n]
    real_words = [(w, v) for w, v in word_shap if v < 0][:top_n]

    # Normalize shap values to 0–1 for opacity scaling
    max_fake = max((v for _, v in fake_words), default=1)
    max_real = max((abs(v) for _, v in real_words), default=1)

    fake_set = {w: v / max_fake for w, v in fake_words}
    real_set = {w: abs(v) / max_real for w, v in real_words}

    html = build_highlight_html(cleaned, fake_set, real_set)

    return {
        "fake_words": fake_words,
        "real_words": real_words,
        "html":       html
    }


def build_highlight_html(text: str, fake_set: dict, real_set: dict) -> str:
    """
    Walks through each word in the cleaned text and wraps
    it in a <span> with background color scaled by SHAP intensity.

    Red highlight  = word pushed prediction toward FAKE
    Green highlight = word pushed prediction toward REAL
    """
    words  = text.split()
    tokens = []

    for word in words:
        # Strip punctuation for lookup but keep original for display
        lookup = word.lower().strip(".,!?;:\"'()[]{}")

        if lookup in fake_set:
            intensity = fake_set[lookup]
            alpha     = round(0.2 + 0.6 * intensity, 2)   # 0.2 – 0.8
            span = (
                f'<span class="shap-fake" '
                f'style="background:rgba(226,75,74,{alpha});'
                f'border-radius:3px;padding:1px 3px;" '
                f'title="Pushes toward FAKE (score: {intensity:.2f})">'
                f'{word}</span>'
            )
            tokens.append(span)

        elif lookup in real_set:
            intensity = real_set[lookup]
            alpha     = round(0.2 + 0.6 * intensity, 2)
            span = (
                f'<span class="shap-real" '
                f'style="background:rgba(99,153,34,{alpha});'
                f'border-radius:3px;padding:1px 3px;" '
                f'title="Pushes toward REAL (score: {intensity:.2f})">'
                f'{word}</span>'
            )
            tokens.append(span)

        else:
            tokens.append(word)

    return " ".join(tokens)