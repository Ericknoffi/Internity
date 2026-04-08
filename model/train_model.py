import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# ── 1. Load dataset ──────────────────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
# Place the CSV at: fake_job_detector/data/fake_jobs.csv

df = pd.read_csv("data/fake_jobs.csv")

# ── 2. Combine text columns into one feature ─────────────────────────────────
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
df["combined_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

# ── 3. Target column ─────────────────────────────────────────────────────────
X = df["combined_text"]
y = df["fraudulent"]   # 0 = real, 1 = fake

# ── 4. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. TF-IDF Vectorizer ─────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),   # unigrams + bigrams
    stop_words="english",
    sublinear_tf=True
)

# ── 6. Train Logistic Regression ─────────────────────────────────────────────
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
lr_model.fit(X_train_tfidf, y_train)

lr_preds = lr_model.predict(X_test_tfidf)
print("=== Logistic Regression ===")
print(classification_report(y_test, lr_preds))

# ── 7. Train Naive Bayes ──────────────────────────────────────────────────────
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_tfidf, y_train)

nb_preds = nb_model.predict(X_test_tfidf)
print("=== Naive Bayes ===")
print(classification_report(y_test, nb_preds))

# ── 8. Save everything ────────────────────────────────────────────────────────
os.makedirs("saved_model", exist_ok=True)
joblib.dump(vectorizer, "saved_model/tfidf_vectorizer.pkl")
joblib.dump(lr_model,   "saved_model/lr_model.pkl")
joblib.dump(nb_model,   "saved_model/nb_model.pkl")

print("\nModels saved to saved_model/")