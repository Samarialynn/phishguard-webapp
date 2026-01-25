
# PhishGuard — Merged Web App

This project merges your training/evaluation scripts into a runnable **web app**:
- **FastAPI backend** (predicts with your saved artifacts or a fallback heuristic)
- **Simple HTML frontend** calling `/api/predict`

## Structure
```text
phishguard_webapp/
├── backend/
│   ├── app.py                # FastAPI server
│   ├── phish_model.py        # Loads artifacts or uses heuristic
│   ├── models/               # Put your trained .joblib files here
│   └── requirements.txt
├── frontend/
│   └── index.html            # Minimal UI
├── legacy/
│   ├── LocalLLM_model.py     # Your original file (kept intact)
│   └── Train_Local_Models_Save_Artifacts.py
└── README.md
```

## How to run locally
1. **Backend**
   ```bash
   cd backend
   python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
   pip install -r requirements.txt
   python app.py
   ```
   The API will run on http://localhost:8000

2. **Frontend**
   Just open `frontend/index.html` in your browser. It will call the backend at `http://localhost:8000` by default when running locally.

## (Optional) Use your trained models
If you already have artifacts from your `Train_Local_Models_Save_Artifacts.py` script, copy them into:
```
phishguard_webapp/backend/models/
```
Expected names (you can adjust in `phish_model.py`):
- `tfidf_vectorizer.joblib`
- `logreg_classifier.joblib`

If these files are present, the API uses them. Otherwise, it falls back to a rule-based heuristic.

## Endpoints
- `GET /api/health` — simple health check
- `POST /api/predict` — body: `{ "text": "..." }` → response: `{ label, confidence, source }`

## Notes
- Your original scripts remain in `legacy/`. You can adapt `Train_Local_Models_Save_Artifacts.py` to save the artifacts into `backend/models/`, or change the paths in that script accordingly.
- If you want to serve the frontend from FastAPI too, you can add a static files mount later.
