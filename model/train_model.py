import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)

def train_and_evaluate():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fake_jobs.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place fake_jobs.csv in data/")

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df['fraudulent'].sum()} fraudulent postings.")

    # 1. Combine text columns
    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    df["combined_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

    X = df["combined_text"]
    y = df["fraudulent"]

    # 2. Train / Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

    # 3. Sublinear Multi-Gram TF-IDF Vectorizer (15,000 features)
    print("\nFitting Sublinear Multi-Gram TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english"
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    # 4. Primary Model: Cost-sensitive Logistic Regression
    print("\nTraining Primary Model: Cost-sensitive Logistic Regression (C=2.5, class_weight={0: 1.0, 1: 8.0})...")
    lr_model = LogisticRegression(
        C=2.5,
        class_weight={0: 1.0, 1: 8.0},
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train_tfidf, y_train)

    lr_probs = lr_model.predict_proba(X_test_tfidf)[:, 1]
    lr_preds = lr_model.predict(X_test_tfidf)
    print("=== Primary Model: Logistic Regression ===")
    print(classification_report(y_test, lr_preds, digits=4))

    # 5. Secondary Model: Calibrated SGDClassifier with Modified Huber Loss
    print("\nTraining Secondary Model: Calibrated SGDClassifier (loss='modified_huber', alpha=1e-5)...")
    sgd_model = SGDClassifier(
        loss="modified_huber",
        alpha=1e-5,
        max_iter=2000,
        random_state=42
    )
    sgd_model.fit(X_train_tfidf, y_train)

    sgd_probs = sgd_model.predict_proba(X_test_tfidf)[:, 1]
    sgd_preds = sgd_model.predict(X_test_tfidf)
    print("=== Secondary Model: SGDClassifier ===")
    print(classification_report(y_test, sgd_preds, digits=4))

    # 6. Calibrated Dual-Model Ensemble Evaluation (65% LR + 35% SGD)
    ensemble_probs = 0.65 * lr_probs + 0.35 * sgd_probs
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, ensemble_probs)
    precisions, recalls, _ = precision_recall_curve(y_test, ensemble_probs)
    pr_auc = auc(recalls, precisions)

    fake_precision = precision_score(y_test, ensemble_preds, pos_label=1)
    fake_recall = recall_score(y_test, ensemble_preds, pos_label=1)
    fake_f1 = f1_score(y_test, ensemble_preds, pos_label=1)
    overall_acc = accuracy_score(y_test, ensemble_preds)

    print("=== Calibrated Dual-Model Ensemble (65% LR + 35% SGD) ===")
    print(classification_report(y_test, ensemble_preds, digits=4))
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Fake Job Precision: {fake_precision:.4f}")
    print(f"Fake Job Recall:    {fake_recall:.4f}")
    print(f"Fake Job F1-Score:  {fake_f1:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"PR-AUC:             {pr_auc:.4f}")

    # 7. Save Artifacts & Metadata
    saved_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_model")
    os.makedirs(saved_dir, exist_ok=True)

    print(f"\nSaving model artifacts to {saved_dir}...")
    joblib.dump(vectorizer, os.path.join(saved_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(lr_model, os.path.join(saved_dir, "lr_model.pkl"))
    joblib.dump(sgd_model, os.path.join(saved_dir, "nb_model.pkl"))  # Kept as nb_model.pkl for seamless drop-in
    joblib.dump(sgd_model, os.path.join(saved_dir, "sgd_model.pkl"))

    metadata = {
        "model_version": "2.0.0",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "Kaggle Real or Fake JobPosting Prediction (17,880 samples)",
        "vectorizer": {
            "type": "TfidfVectorizer",
            "max_features": 15000,
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
            "stop_words": "english"
        },
        "primary_model": {
            "type": "LogisticRegression",
            "C": 2.5,
            "class_weight": {"0": 1.0, "1": 8.0},
            "weight_in_ensemble": 0.65
        },
        "secondary_model": {
            "type": "SGDClassifier",
            "loss": "modified_huber",
            "alpha": 1e-5,
            "weight_in_ensemble": 0.35
        },
        "benchmark_metrics": {
            "test_split": 0.2,
            "test_samples": len(y_test),
            "fake_precision": round(fake_precision, 4),
            "fake_recall": round(fake_recall, 4),
            "fake_f1_score": round(fake_f1, 4),
            "accuracy": round(overall_acc, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4)
        }
    }

    metadata_path = os.path.join(saved_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts and metadata saved successfully: {metadata_path}")

if __name__ == "__main__":
    train_and_evaluate()