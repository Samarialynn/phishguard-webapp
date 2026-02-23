from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import re
import datetime

try:
    from joblib import load
except Exception:
    load = None

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
VEC_PATH = MODELS_DIR / "vectorizer.joblib"
CLF_PATH = MODELS_DIR / "classifier.joblib"

vectorizer = None
classifier = None


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    mode: str = Field("email")


# ==========================================
# ADVANCED HEURISTIC PHISHING DETECTION
# ==========================================
def extract_signals(text: str):
    t = (text or "").lower()
    reasons = []
    score = 0.0

    # URL detection
    urls = re.findall(r"(https?://\S+|www\.\S+)", text, flags=re.IGNORECASE)
    if urls:
        score += 0.25
        reasons.append(f"Contains link(s): {min(len(urls),3)} detected.")

    # Brand impersonation + login/update language
    if re.search(r"(paypal|bank|amazon|apple|microsoft|netflix|chase|wellsfargo)", t):
        if re.search(r"(verify|login|secure|account|update|confirm)", t):
            score += 0.40
            reasons.append("Impersonates trusted brand with login/verify language.")

    # Suspicious TLD
    if re.search(r"\.(ru|cn|tk|ml|top|xyz|click|gq|work)$", t):
        score += 0.25
        reasons.append("Suspicious top-level domain.")

    # Multiple hyphens (common phishing domain pattern)
    if text.count("-") >= 2:
        score += 0.20
        reasons.append("Domain contains multiple hyphens (common phishing pattern).")

    # Urgency / time pressure
    urgency = [
        "urgent", "immediately", "act now",
        "before it expires", "limited time",
        "final notice", "suspended", "locked"
    ]
    if any(u in t for u in urgency):
        score += 0.30
        reasons.append("Uses urgency or time-pressure language.")

    # Credential / payment language
    creds = [
        "password", "verify", "login", "sign in",
        "bank", "invoice", "payment", "gift card",
        "confirm identity", "update billing"
    ]
    if any(c in t for c in creds):
        score += 0.30
        reasons.append("Requests sensitive credentials or financial action.")

    # Crypto scam detection
    if re.search(r"(btc|bitcoin|crypto|eth|usdt)", t):
        score += 0.35
        reasons.append("Mentions cryptocurrency reward (common scam pattern).")

    # Monetary lure ($ amounts)
    if re.search(r"\$\d+", t):
        score += 0.30
        reasons.append("Contains monetary incentive.")

    # Reward bait
    bait = [
        "you won", "prize", "free",
        "reward", "claim", "congratulations",
        "selected to receive"
    ]
    if any(b in t for b in bait):
        score += 0.30
        reasons.append("Contains reward/prize bait language.")

    score = min(score, 0.95)
    return score, reasons, urls[:10]


# ==========================================
# ML MODEL SUPPORT (optional)
# ==========================================
def ml_predict_proba(text: str):
    if not vectorizer or not classifier:
        return None

    X = vectorizer.transform([text])
    proba = classifier.predict_proba(X)[0]
    classes = list(classifier.classes_)

    if "phishing" in classes:
        return float(proba[classes.index("phishing")])

    return float(max(proba))


# ==========================================
# LOAD ML MODELS
# ==========================================
@app.on_event("startup")
def load_models():
    global vectorizer, classifier
    if load and VEC_PATH.exists() and CLF_PATH.exists():
        vectorizer = load(VEC_PATH)
        classifier = load(CLF_PATH)


# ==========================================
# HEALTH CHECK
# ==========================================
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(vectorizer and classifier)
    }


# ==========================================
# MAIN PREDICTION ENDPOINT
# ==========================================
@app.post("/api/predict")
def predict(payload: PredictIn):
    text = payload.text.strip()

    h_score, reasons, urls = extract_signals(text)
    p_ml = ml_predict_proba(text)

    if p_ml is None:
        confidence = h_score
        source = "heuristic"
    else:
        confidence = min(0.80 * p_ml + 0.20 * h_score, 1.0)
        source = "ml+heuristic"

    # Boost SMS sensitivity slightly
    if payload.mode == "sms":
        confidence = min(confidence + 0.05, 1.0)

    # Lower threshold for better real-world detection
    label = "phishing" if confidence >= 0.40 else "legitimate"

    # Log detection (for realism)
    print({
        "timestamp": str(datetime.datetime.utcnow()),
        "mode": payload.mode,
        "confidence": confidence,
        "label": label
    })

    return {
        "label": label,
        "confidence": confidence,
        "source": source,
        "mode": payload.mode,
        "reasons": reasons[:5],
        "urls_found": urls
    }