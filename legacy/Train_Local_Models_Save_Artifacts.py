# === TRAINING SCRIPT: Save TF-IDF + Logistic Regression, SVM, Random Forest ===
# Run this ONCE to create your local model artifacts for later TEST-ONLY use.

import os, re
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG ----------------
SEED = 42
TEST_SIZE = 0.20

DATA_PATH = "C:/Users/SMASH Scholar/Desktop/phishing-email-detector/data/combined_data.csv"
ARTIFACT_DIR = "C:/Users/SMASH Scholar/Desktop/phishing-email-detector/artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

VECT_PATH   = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.joblib")
LOGREG_PATH = os.path.join(ARTIFACT_DIR, "logreg.joblib")
SVM_PATH    = os.path.join(ARTIFACT_DIR, "svm_calibrated.joblib")
RF_PATH     = os.path.join(ARTIFACT_DIR, "random_forest.joblib")

np.random.seed(SEED)
print(f"[INFO] Training local models and saving to: {ARTIFACT_DIR}")

# ---------------- LOAD DATA ----------------
def infer_columns(df: pd.DataFrame):
    text_cands = [c for c in df.columns if re.search(r'(text|email|content|message|body)', c, re.I)]
    label_cands = [c for c in df.columns if re.search(r'(label|target|class|phishing|is_*phish)', c, re.I)]
    if not text_cands:
        for c in df.columns:
            if df[c].dtype == object:
                text_cands = [c]; break
    if not label_cands:
        for c in df.columns:
            uniq = pd.unique(df[c])
            if len(uniq) == 2:
                label_cands = [c]; break
    if not text_cands or not label_cands:
        raise ValueError(f"Could not infer text/label columns. Found: {df.columns.tolist()}")
    return text_cands[0], label_cands[0]

def normalize_label(v):
    if isinstance(v, (int, np.integer)): return int(v)
    s = str(v).strip().lower()
    if s in {"1","phish","phishing","spam","malicious","true","yes"}: return 1
    if s in {"0","ham","legit","legitimate","benign","false","no"}:  return 0
    return 0

df = pd.read_csv(DATA_PATH)
text_col, label_col = infer_columns(df)
data = df[[text_col, label_col]].dropna().copy()
data["label_bin"] = data[label_col].apply(normalize_label)

X_train, X_test, y_train, y_test = train_test_split(
    data[text_col], data["label_bin"],
    test_size=TEST_SIZE, random_state=SEED, stratify=data["label_bin"]
)

# ---------------- TF-IDF VECTORIZE ----------------
print("[INFO] Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), lowercase=True)
Xtr = vectorizer.fit_transform(X_train)
dump(vectorizer, VECT_PATH)
print(f"[SAVED] TF-IDF vectorizer → {VECT_PATH}")

# ---------------- LOGISTIC REGRESSION ----------------
print("[INFO] Training Logistic Regression...")
logreg = LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=-1)
logreg.fit(Xtr, y_train)
dump(logreg, LOGREG_PATH)
print(f"[SAVED] Logistic Regression → {LOGREG_PATH}")

# ---------------- CALIBRATED LINEAR SVM ----------------
print("[INFO] Training Linear SVM (Calibrated)...")
svm_base = LinearSVC(class_weight="balanced", random_state=SEED)
svm = CalibratedClassifierCV(svm_base, method="sigmoid", cv=3)
svm.fit(Xtr, y_train)
dump(svm, SVM_PATH)
print(f"[SAVED] Calibrated SVM → {SVM_PATH}")

# ---------------- RANDOM FOREST ----------------
print("[INFO] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=SEED,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(Xtr, y_train)
dump(rf, RF_PATH)
print(f"[SAVED] Random Forest → {RF_PATH}")

print("\n✅ All artifacts trained and saved successfully!")
print(f"- TF-IDF vectorizer: {VECT_PATH}")
print(f"- Logistic Regression: {LOGREG_PATH}")
print(f"- Linear SVM: {SVM_PATH}")
print(f"- Random Forest: {RF_PATH}")
