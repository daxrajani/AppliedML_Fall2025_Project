# Health Symptom Analyzer

A production-style machine learning project that performs symptom-based condition triage using an ensemble classifier.  
The repository includes model training, reusable inference logic, a Streamlit application, a FastAPI service, and automated quality checks.

---

## Impact Summary (Recruiter-Friendly)

- Built an end-to-end ML triage system from training to deployment-ready interfaces (web + API).
- Improved prediction trust by introducing top-3 ranked outputs and confidence-based inconclusive handling.
- Increased project reliability by centralizing inference logic, reducing behavior drift across UI/API.
- Added production engineering standards: test coverage for core inference behavior and CI automation.
- Packaged the project with model governance documentation (`MODEL_CARD.md`) for responsible AI communication.

### Outcome Highlights

- Dual-serving interfaces (`Streamlit` for demos, `FastAPI` for integration).
- Shared inference engine used consistently across all runtime surfaces.
- Automated validation pipeline in GitHub Actions for repeatable quality checks.

---

## 1) Project Objective

The goal is to demonstrate a complete applied ML workflow:

- train and evaluate multiple classification models
- aggregate predictions with a weighted soft-voting ensemble
- expose inference through both UI and API surfaces
- enforce confidence-aware response behavior for safer outputs

This project is designed for educational and portfolio use, with engineering practices that mirror real-world ML product development.

---

## 2) System Architecture

### Core components

- `main.py`  
  Trains individual models, builds the ensemble, persists artifacts, and supports CLI predictions.

- `inference.py`  
  Single source of truth for runtime prediction logic: normalization, symptom validation, feature mapping, top-k ranking, and inconclusive threshold handling.

- `app.py`  
  Streamlit frontend for interactive symptom selection and triage output.

- `api.py`  
  FastAPI service exposing machine-consumable endpoints (`/health`, `/symptoms`, `/predict`).

- `scripts/evaluation.py`  
  Offline evaluation script that generates metrics, confusion matrices, and interpretability outputs.

- `models/`  
  Model-specific constructors and configuration helpers.

### Artifacts and generated outputs

- `saved_models_main/`: serialized model artifacts
- `available_symptoms.txt`: valid feature-space symptom names
- `disease_names.txt`: label mapping used for inference
- `evaluation_results/`: quantitative and qualitative evaluation outputs

---

## 3) Runtime Flow

1. User submits symptoms (UI/API/CLI).
2. Symptoms are normalized and deduplicated.
3. Invalid symptoms are tracked and excluded.
4. Input is mapped to the model feature space.
5. Ensemble probabilities are computed.
6. Response includes:
   - top prediction
   - top-3 ranked conditions with confidence
   - inconclusive flag when confidence is below threshold

This design improves transparency and reduces overconfident single-label behavior.

---

## Engineering Notes (Hiring Manager / Technical Depth)

- **Inference consistency:** `inference.py` is the single prediction path for UI and API.
- **Feature mapping safety:** runtime inference aligns with model feature ordering to avoid symptom-vector mismatch.
- **Input robustness:** symptom normalization, deduplication, ignored-token tracking, and minimum-symptom guardrails.
- **Decision policy:** configurable inconclusive threshold to avoid high-confidence claims on weak evidence.
- **Test strategy:** targeted unit tests for core business logic and edge-case behavior.
- **Delivery readiness:** CI checks for test execution and syntax validation on push/PR.

---

## 4) Local Setup

### Prerequisites

- Python 3.11+
- pip

### Installation (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Train or load model artifacts

```powershell
python main.py
```

If model files are already present in `saved_models_main/`, they are loaded; otherwise training is performed and artifacts are saved.

---

## 5) Run the Application

### Streamlit UI

```powershell
streamlit run app.py
```

### FastAPI service

```powershell
uvicorn api:app --reload
```

Default API base URL: `http://127.0.0.1:8000`

---

## 6) API Contract

### `GET /health`
Basic service health check.

### `GET /symptoms`
Returns the valid symptom list aligned with model feature ordering.

### `POST /predict`
Predicts likely conditions from input symptoms.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"symptoms\": [\"high_fever\", \"chills\", \"headache\", \"nausea\"]}"
```

---

## 7) Testing and CI

### Local tests

```powershell
pytest -q tests
```

### CI pipeline

GitHub Actions workflow at `.github/workflows/ci.yml` runs:

- dependency installation
- unit tests
- syntax validation (`compileall`)

The workflow is configured for stable test discovery and modern action runtime compatibility.

---

## 8) Evaluation and Interpretability

Run:

```powershell
python scripts/evaluation.py
```

The script generates model comparison outputs under `evaluation_results/`, including:

- comparative metrics
- confusion matrices
- report files
- interpretability artifacts (SHAP/LIME where applicable)

---

## 9) Responsible Use

This repository is an educational triage prototype and is **not** a diagnostic medical device.  
Predictions should never replace clinical assessment by qualified professionals.

For intended use, limitations, and risk notes, see `MODEL_CARD.md`.

---

## 10) Team

- Dax Rajani
- Harsh Ahuja
- Charanish Miriyala
