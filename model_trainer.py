import os
import re
import json
import time
import joblib
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_wire_and_agency_artifacts(text):
    """
    Strips wire agency datelines (e.g. 'WASHINGTON (Reuters) -') and
    boilerplate photo credits ('image via ...') to prevent models from
    overfitting to publisher signatures.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove leading datelines like "WASHINGTON (Reuters) -" or "SEATTLE/WASHINGTON (Reuters) -"
    text = re.sub(r'^[A-Za-z\s,/.]+\s*\((?:Reuters|REUTERS|AP|AFP)\)\s*[-—]\s*', '', text)
    # Remove standalone agency mentions and photo attribution tags
    text = re.sub(r'\b(?:reuters|Reuters|REUTERS)\b', '', text)
    text = re.sub(r'\b(?:image via|featured image via|photo by|photo credit)[^\n.]*', '', text, flags=re.I)
    
    return text

def clean_text_for_nlp(text):
    """
    Normalizes text for TF-IDF feature extraction:
    lower-case, URL removal, HTML removal, punctuation stripping, and whitespace normalization.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def train_and_evaluate_models():
    print("=" * 60)
    print("TruthLens: De-biased Fraud & Fake News Model Training Pipeline")
    print("=" * 60)

    start_time = time.time()
    
    fake_path = "archive/Fake.csv"
    true_path = "archive/True.csv"

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError("Dataset files archive/Fake.csv and archive/True.csv are required.")

    print("\n[1/6] Loading datasets...")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    print(f"  - Loaded Fake news samples: {len(fake_df):,}")
    print(f"  - Loaded Real news samples: {len(true_df):,}")

    fake_df["label"] = 1  # 1 = Fake / Fraudulent
    true_df["label"] = 0  # 0 = Real / Legitimate

    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("\n[2/6] De-biasing & cleaning raw news text...")
    print("  - Removing wire datelines, publisher artifacts (e.g. 'Reuters' leakage)...")
    df["title_clean"] = df["title"].fillna("").apply(clean_wire_and_agency_artifacts)
    df["text_clean"] = df["text"].fillna("").apply(clean_wire_and_agency_artifacts)

    # Combine Title and Text
    df["full_content"] = df["title_clean"] + " " + df["text_clean"]

    print("  - Performing NLP normalization (lowercasing, URL/HTML stripping, tokenizing)...")
    df["clean_content"] = df["full_content"].apply(clean_text_for_nlp)

    X = df["clean_content"]
    y = df["label"]

    print(f"  - Total preprocessed samples: {len(X):,}")

    print("\n[3/6] Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  - Training samples: {len(X_train):,}")
    print(f"  - Testing samples:  {len(X_test):,}")

    print("\n[4/6] Generating TF-IDF Features (unigrams + bigrams, sublinear TF)...")
    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"  - TF-IDF Matrix shape: {X_train_tfidf.shape}")

    models_to_train = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=2.0, random_state=42),
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000, C=0.5, random_state=42)
    }

    trained_models = {}
    metrics_summary = {}

    print("\n[5/6] Training and evaluating multi-model ensemble...")
    for name, model in models_to_train.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        model.fit(X_train_tfidf, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test_tfidf)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()

        trained_models[name] = model
        metrics_summary[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "training_time_sec": round(train_time, 2)
        }

        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall:    {rec * 100:.2f}%")
        print(f"  F1-Score:  {f1 * 100:.2f}%")
        print("  Confusion Matrix (Real=0, Fake=1):", cm)

    print("\n[6/6] Verifying de-biasing with non-Reuters benchmarks...")
    real_sample = "The Federal Reserve held benchmark interest rates steady on Wednesday following a two-day monetary policy meeting, citing balanced inflation progress."
    fake_sample = "SHOCKING SECRET: Miracle investment loop gives you 500% guaranteed profit overnight! Act now before Wall Street shuts it down!"

    lr_model = trained_models["LogisticRegression"]
    real_vec = vectorizer.transform([clean_text_for_nlp(real_sample)])
    fake_vec = vectorizer.transform([clean_text_for_nlp(fake_sample)])

    real_pred = lr_model.predict(real_vec)[0]
    real_prob = lr_model.predict_proba(real_vec)[0]
    fake_pred = lr_model.predict(fake_vec)[0]
    fake_prob = lr_model.predict_proba(fake_vec)[0]

    print(f"  Benchmark Real News (No Reuters tag): {'REAL (PASS)' if real_pred == 0 else 'FAKE (FAIL)'} [Real: {real_prob[0]*100:.1f}%, Fake: {real_prob[1]*100:.1f}%]")
    print(f"  Benchmark Fake News:                 {'FAKE (PASS)' if fake_pred == 1 else 'REAL (FAIL)'} [Real: {fake_prob[0]*100:.1f}%, Fake: {fake_prob[1]*100:.1f}%]")

    # Save artifacts
    print("\nSaving trained models and vectorizer to disk...")
    joblib.dump(lr_model, "model.pkl")
    joblib.dump(trained_models["MultinomialNB"], "nb_model.pkl")
    joblib.dump(trained_models["PassiveAggressive"], "pac_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    with open("model_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    total_time = time.time() - start_time
    print(f"\nAll models and artifacts saved successfully in {total_time:.2f} seconds!")

if __name__ == "__main__":
    train_and_evaluate_models()
