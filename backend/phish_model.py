
import os
from pathlib import Path
from typing import Dict, Any
try:
    from joblib import load
except Exception:
    load = None  # If joblib isn't installed yet

import re

MODELS_DIR = Path(__file__).parent / "models"
VECT_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
CLF_PATH  = MODELS_DIR / "logreg_classifier.joblib"

_vectorizer = None
_classifier = None

def _try_load_models():
    global _vectorizer, _classifier
    if load is None:
        return False
    if VECT_PATH.exists() and CLF_PATH.exists():
        try:
            _vectorizer = load(VECT_PATH)
            _classifier = load(CLF_PATH)
            return True
        except Exception:
            return False
    return False

def _heuristic_score(text: str) -> float:
    """Very small fallback when artifacts are missing.

    Returns a phishing confidence in [0,1]."""
    t = (text or "").lower()
    signals = [
        r"verify your account",
        r"suspended",
        r"click (here|link)",
        r"urgent",
        r"password",
        r"reset",
        r"confirm information",
        r"bank|invoice|wire|gift card",
        r"http[s]?://[\w./-]+"
    ]
    score = 0
    for s in signals:
        if re.search(s, t):
            score += 1
    # Normalize: 0 to ~1 range (cap at 1.0)
    return min(score / 5.0, 1.0)

def predict(text: str) -> Dict[str, Any]:
    """Return dict with: {label: 'phishing'|'legitimate', confidence: float, source: 'artifacts'|'heuristic'}"""
    # Try models
    if _vectorizer is None or _classifier is None:
        _try_load_models()

    if _vectorizer is not None and _classifier is not None:
        # Use trained artifacts (preferred)
        X = _vectorizer.transform([text or ""])  # probabilities if available
        try:
            # LogisticRegression has predict_proba
            probs = _classifier.predict_proba(X)[0]
            phish_prob = float(probs[1])
        except Exception:
            # For models without predict_proba (e.g., LinearSVC), synthesize confidence
            pred = int(_classifier.predict(X)[0])
            phish_prob = 0.75 if pred == 1 else 0.25

        label = "phishing" if phish_prob >= 0.5 else "legitimate"
        return {"label": label, "confidence": phish_prob, "source": "artifacts"}

    # Fallback: heuristic
    phish_prob = _heuristic_score(text or "")
    label = "phishing" if phish_prob >= 0.5 else "legitimate"
    return {"label": label, "confidence": phish_prob, "source": "heuristic"}
