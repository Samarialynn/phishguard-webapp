import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "training_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def main():
    print(f"Loading data from: {DATA_PATH}")
    if not DATA_PATH.exists():
        print("ERROR: training_data.csv not found. Needs columns: text,label")
        return

    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    # Optional: enforce only 2 labels
    df = df[df["label"].isin(["phishing", "legitimate"])]

    if df.empty:
        print("ERROR: No usable rows. Labels must be phishing/legitimate.")
        return

    X = df["text"].tolist()
    y = df["label"].tolist()

    print(f"Loaded {len(X)} examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=300, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    print(classification_report(y_test, preds))

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test_vec)
        classes = list(clf.classes_)
        if "phishing" in classes:
            y_bin = [1 if yy == "phishing" else 0 for yy in y_test]
            auc = roc_auc_score(y_bin, proba[:, classes.index("phishing")])
            print("ROC-AUC:", round(auc, 3))

    dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
    dump(clf, MODELS_DIR / "classifier.joblib")
    print("
