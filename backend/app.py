from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import re

try:
    from joblib import load
except Exception:
    load = None

app = FastAPI()

# CORS (allow frontend access)
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


# =========================
# HEURISTIC DETECTION LOGIC
# =========================
def extract_signals(text: str):
    t = (text or "").lower()
    reasons = []
    score = 0.0

    # URL detection
    urls = re.findall(r"(https?://\S+|www\.\S+)", text, flags=re.IGNORECASE)
    if urls:
        score += 0.20
        reasons.append(f"Contains link(s): {min(len(urls),3)} detected.")

    # Brand impersonation + login language
    if re.search(r"(paypal|bank|amazon|apple|microsoft).*(verify|login|secure|account)", t):
        score += 0.35
        reasons.append("Impersonates trusted brand with login/verify language.")

    # Multiple hyphens (very common phishing trick)
    if text.count("-") >= 2:
        score += 0.15
        reasons.append("Domain contains multiple hyphens (common phishing pattern).")

    # Suspicious TLD
    if re.search(r"\.(ru|cn|tk|ml|top|xyz)$", t):
        score += 0.20
        reasons.append("Suspicious top-level domain.")

    # Urgency / fear tactics
    urgency = [
        "urgent", "immediately", "act now", "account suspended",
        "locked", "final notice", "verify now", "limited time"
    ]
    if any(u in t for u in urgency):
        score += 0.25
        reasons.append("Uses urgency or threat language.")

    # Credential / money request
    creds = [
        "password", "verify", "login", "sign in", "reset",
        "bank", "invoice", "payment", "gift card", "crypto"
    ]
    if any(c in t for c in creds):
        score += 0.25
        reasons.append("Asks for credentials or money-related action.")

    # Reward bait
    bait = ["you won", "prize", "free", "reward", "claim", "congratulations"]
    if any(b in t for b in bait):
        score += 0.15
        reasons.append("Contains reward/prize bait language.")

    # Social engineering cues
    if "reply-to" in t or "sent from my iphone" in t:
        score += 0.05
        reasons.append("Contains common social-engineering sender cues.")

    score = min(score, 0.95)
    return score, reasons, urls[:10]


# =========================
# ML MODEL PREDICTION
# =========================
def ml_predict_proba(text: str):
    if not vectorizer or not classifier:
        return None

    X = vectorizer.transform([text])
    proba = classifier.predict_proba(X)[0]
    classes = list(classifier.classes_)

    if "phishing" in classes:
        return float(proba[classes.index("phishing")])

    return float(max(proba))


# =========================
# LOAD MODELS ON STARTUP
# =========================
@app.on_event("startup")
def load_models():
    global vectorizer, classifier
    if load and VEC_PATH.exists() and CLF_PATH.exists():
        vectorizer = load(VEC_PATH)
        classifier = load(CLF_PATH)


# =========================
# HEALTH CHECK
# =========================
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(vectorizer and classifier)
    }


# =========================
# MAIN PREDICTION ENDPOINT
# =========================
@app.post("/api/predict")
def predict(payload: PredictIn):
    text = payload.text.strip()

    h_score, reasons, urls = extract_signals(text)
    p_ml = ml_predict_proba(text)

    if p_ml is None:
        confidence = h_score
        label = "phishing" if confidence >= 0.5 else "legitimate"
        source = "heuristic"
    else:
        confidence = min(0.85 * p_ml + 0.15 * h_score, 1.0)
        label = "phishing" if confidence >= 0.5 else "legitimate"
        source = "ml+heuristic"

    return {
        "label": label,
        "confidence": confidence,
        "source": source,
        "mode": payload.mode,
        "reasons": reasons[:5],
        "urls_found": urls
    }