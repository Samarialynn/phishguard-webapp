# === Phishing Email Detection: Gemini 2.0 Flash vs Local TF-IDF + Logistic Regression ===
# Metrics: Accuracy, Precision, Recall/TPR, F1, FPR, ROC-AUC + Confusion Matrices
# Dataset: C:/Users/SMASH Scholar/Desktop/phishing-email-detector/data/combined_data.csv

import os, re, json, time
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
# If you don't want to call Gemini at all, keep DRY_RUN=True (it will use a heuristic instead).
DRY_RUN = False                 # <<< set to False ONLY if you have GOOGLE_API_KEY and want real Gemini calls
OVERWRITE_PREDICTIONS = True   # overwrite cached predictions
SEED = 42
TEST_SIZE = 0.20

# Models/paths
GEMINI_MODEL = "gemini-2.0-flash"
DATA_PATH = "C:/Users/SMASH Scholar/Desktop/phishing-email-detector/data/combined_data.csv"

OUT_DIR = "C:/Users/SMASH Scholar/Desktop/phishing-email-detector"
GEMINI_PRED_PATH = os.path.join(OUT_DIR, "gemini_preds.csv")
LOCAL_PRED_PATH  = os.path.join(OUT_DIR, "local_preds.csv")
REPORT_PATH      = os.path.join(OUT_DIR, "llm_vs_local_report.csv")
EVAL_EXPORT      = os.path.join(OUT_DIR, "eval_aligned_rows.csv")

np.random.seed(SEED)

print(f"[INFO] DRY_RUN={DRY_RUN}  OVERWRITE_PREDICTIONS={OVERWRITE_PREDICTIONS}")
print(f"[INFO] DATA: {DATA_PATH}")

# ---------------- HELPERS ----------------
def infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
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

def normalize_label(v) -> int:
    if isinstance(v, (int, np.integer)): return int(v)
    s = str(v).strip().lower()
    if s in {"1","phish","phishing","spam","malicious","true","yes"}: return 1
    if s in {"0","ham","legit","legitimate","benign","false","no"}:  return 0
    return 0  # default: legit

def format_prompt(email_text: str) -> str:
    return (
        "You are a strict email security classifier. "
        "Read the email and respond ONLY with one JSON line like:\n"
        '{"label": "<phishing|legitimate>", "confidence": <0.0-1.0>}\n\n'
        f"Email:\n{email_text}\n\n"
        "Rules:\n- No extra text or markdown.\n- Label must be 'phishing' or 'legitimate'.\n"
        "- Confidence = your certainty from 0 to 1."
    )

def parse_llm_json(s: str) -> Dict[str, Any]:
    try:
        d = json.loads(s.strip())
        label = str(d.get("label","")).lower()
        conf = float(d.get("confidence", 0.5))
        if label not in {"phishing","legitimate"}:
            label = "phishing" if conf >= 0.5 else "legitimate"
        conf = max(0.0, min(1.0, conf))
        return {"label": label, "confidence": conf}
    except Exception:
        return {"label": "legitimate", "confidence": 0.5}

def heuristic_baseline(text: str) -> Dict[str, Any]:
    pats = [r"verify your account", r"urgent", r"click here", r"http", r"https",
            r"reset your password", r"bank", r"wire transfer", r"invoice attached",
            r"security alert", r"login", r"password", r"confirm.*(identity|account)"]
    score = 0.1 + 0.15*sum(bool(re.search(p, text, re.I)) for p in pats)
    score = min(0.95, score)
    return {"label": "phishing" if score >= 0.5 else "legitimate", "confidence": float(score)}

# ---------------- GEMINI (optional) ----------------
def call_gemini(text: str, model: str = GEMINI_MODEL) -> Dict[str, Any]:
    if DRY_RUN:
        return heuristic_baseline(text)
    import google.generativeai as genai
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_API_KEY (set it or keep DRY_RUN=True).")
    genai.configure(api_key=key)
    resp = genai.GenerativeModel(model).generate_content(format_prompt(text))
    out = getattr(resp, "text", "") or ""
    return parse_llm_json(out)

def batched_predict_gemini(texts: pd.Series, cache_path: str) -> pd.DataFrame:
    if (not OVERWRITE_PREDICTIONS) and os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    rows = []
    for i, t in enumerate(texts):
        try:
            res = call_gemini(str(t))
        except Exception:
            res = {"label":"legitimate","confidence":0.5}
        rows.append({"idx": i, "pred_label": res["label"], "pred_conf": res["confidence"]})
        if not DRY_RUN:
            time.sleep(0.15)  # gentle pacing for API
    out = pd.DataFrame(rows)
    out.to_csv(cache_path, index=False)
    return out

# ---------------- LOCAL BASELINE (free) ----------------
def run_local_baseline(X_train: pd.Series, y_train: pd.Series, X_test: pd.Series):
    """
    Free local classifier: TF-IDF (1-2 grams) + Logistic Regression
    Returns: preds (0/1), probs (float)
    """
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), lowercase=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=300, n_jobs=-1)
    clf.fit(Xtr, y_train)

    probs = clf.predict_proba(Xte)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def save_local_preds(preds: np.ndarray, probs: np.ndarray, cache_path: str):
    dfp = pd.DataFrame({"pred_label": np.where(preds==1, "phishing", "legitimate"),
                        "pred_conf": probs})
    dfp.to_csv(cache_path, index=False)
    return dfp

# ---------------- METRICS / PLOTTING ----------------
def evaluate(y_true, y_pred, conf=None) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)  # = TPR
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr = fp / (fp + tn) if (fp+tn)>0 else 0.0
    res = {"Accuracy":acc,"Precision":prec,"Recall_TPR":rec,"F1":f1,"FPR":fpr,
           "TP":tp,"FP":fp,"TN":tn,"FN":fn}
    if conf is not None and len(np.unique(y_true))==2:
        try: res["ROC_AUC"] = roc_auc_score(y_true, conf)
        except Exception: res["ROC_AUC"] = np.nan
    else:
        res["ROC_AUC"] = np.nan
    return res

def plot_metric_bar(df: pd.DataFrame, metric: str):
    plt.figure(figsize=(8,4))
    plt.bar(df["Model"], df[metric])
    plt.ylabel(metric); plt.title(f"{metric} by Model")
    plt.xticks(rotation=15); plt.tight_layout(); plt.show()

def plot_confusion(cm: np.ndarray, title: str):
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks([0,1], ["Legitimate","Phishing"]); plt.yticks([0,1], ["Legitimate","Phishing"])
    for (i,j),z in np.ndenumerate(cm): plt.text(j,i,str(int(z)),ha='center',va='center')
    plt.tight_layout(); plt.show()

def plot_roc(y_true, conf, title):
    try:
        fpr, tpr, _ = roc_curve(y_true, conf)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr); plt.plot([0,1],[0,1], linestyle='--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(title)
        plt.tight_layout(); plt.show()
    except Exception:
        pass

# ---------------- MAIN ----------------
# Load data
df = pd.read_csv(DATA_PATH)
text_col, label_col = infer_columns(df)
data = df[[text_col, label_col]].dropna().copy()
data["label_bin"] = data[label_col].apply(normalize_label)

X_train, X_test, y_train, y_test = train_test_split(
    data[text_col], data["label_bin"],
    test_size=TEST_SIZE, random_state=SEED, stratify=data["label_bin"]
)

# --- Gemini predictions (heuristic if DRY_RUN=True) ---
gem_df = batched_predict_gemini(X_test, GEMINI_PRED_PATH)
gem_pred = (gem_df["pred_label"].str.lower()=="phishing").astype(int).values
gem_conf = gem_df["pred_conf"].values

# --- Local baseline predictions (always free) ---
loc_pred, loc_conf = run_local_baseline(X_train, y_train, X_test)
loc_df = save_local_preds(loc_pred, loc_conf, LOCAL_PRED_PATH)

# Align everything into one table
eval_df = pd.DataFrame({
    "text": X_test.values,
    "y_true": y_test.values,
    "gemini_label": gem_df["pred_label"].values,
    "gemini_conf":  gem_conf,
    "local_label":  np.where(loc_pred==1, "phishing", "legitimate"),
    "local_conf":   loc_conf,
})
eval_df["gemini_pred"] = (eval_df["gemini_label"].str.lower()=="phishing").astype(int)
eval_df["local_pred"]  = (eval_df["local_label"].str.lower()=="phishing").astype(int)
eval_df.to_csv(EVAL_EXPORT, index=False)

# Metrics
gem_metrics = evaluate(eval_df["y_true"].values, eval_df["gemini_pred"].values, eval_df["gemini_conf"].values)
loc_metrics = evaluate(eval_df["y_true"].values, eval_df["local_pred"].values,  eval_df["local_conf"].values)

report = pd.DataFrame([
    {"Model":"Gemini 2.0 Flash (DRY heuristic)" if DRY_RUN else "Gemini 2.0 Flash", **gem_metrics},
    {"Model":"Local TF-IDF + Logistic Regression", **loc_metrics},
])
report.to_csv(REPORT_PATH, index=False)
print(report)

# Visuals
for m in ["Accuracy","Precision","Recall_TPR","F1"]:
    plot_metric_bar(report, m)

plot_confusion(confusion_matrix(eval_df["y_true"], eval_df["gemini_pred"], labels=[0,1]),
               "Gemini — Confusion Matrix")
plot_confusion(confusion_matrix(eval_df["y_true"], eval_df["local_pred"], labels=[0,1]),
               "Local TF-IDF + Logistic Regression — Confusion Matrix")

plot_roc(eval_df["y_true"], eval_df["gemini_conf"], "Gemini — ROC")
plot_roc(eval_df["y_true"], eval_df["local_conf"], "Local TF-IDF + Logistic Regression — ROC")

print("\nArtifacts:")
print(f"- Report: {REPORT_PATH}")
print(f"- Eval rows: {EVAL_EXPORT}")
print(f"- Gemini preds: {GEMINI_PRED_PATH}")
print(f"- Local preds: {LOCAL_PRED_PATH}")
