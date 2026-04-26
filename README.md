# Health Symptom Analyzer

A symptom-based multi-class classification project built with scikit-learn/XGBoost, exposed through both Streamlit and FastAPI runtimes.

---

## Scope

- Training and persistence of individual classifiers and a weighted soft-voting ensemble
- Shared inference layer used by both UI and API paths
- Confidence-ranked predictions (top-3) with inconclusive decision threshold
- Automated tests and CI checks

---

## Repository Layout

- `main.py`: model training/loading and CLI inference loop
- `inference.py`: normalization, validation, feature mapping, probability ranking
- `app.py`: Streamlit interface
- `api.py`: FastAPI service
- `models/`: model factory functions and tuned configuration
- `scripts/evaluation.py`: offline evaluation and artifact generation
- `tests/`: unit tests for inference behavior
- `saved_models_main/`: serialized model artifacts
- `evaluation_results/`: generated evaluation outputs

---

## Data and Artifacts

- Input training file: `Prototype.csv`
- Generated symptom vocabulary: `available_symptoms.txt`
- Generated class labels: `disease_names.txt`
- Model artifacts: `saved_models_main/*.pkl`

`main.py` regenerates vocabulary/label files and trains missing model files as required.

---

## Inference Behavior

Runtime inference pipeline (`inference.py`):

1. Normalize input symptom tokens
2. Apply synonym mapping where configured
3. Deduplicate and separate invalid symptoms
4. Map to model feature space
5. Run `predict_proba`
6. Return top-3 ranked classes with confidence
7. Mark result as inconclusive when confidence < threshold

The minimum-symptom requirement and inconclusive threshold are configurable constants.

---

## Local Setup (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train/load model artifacts:

```powershell
python main.py
```

---

## Run Targets

Streamlit:

```powershell
streamlit run app.py
```

FastAPI:

```powershell
uvicorn api:app --reload
```

---

## API Endpoints

### `GET /health`
Service health metadata.

### `GET /symptoms`
Returns active symptom feature list used for inference.

### `POST /predict`
Request body:

```json
{
  "symptoms": ["high_fever", "chills", "headache"]
}
```

Response includes:
- accepted/ignored symptoms
- top-3 ranked predictions with confidence
- inconclusive flag

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"symptoms\": [\"high_fever\", \"chills\", \"headache\", \"nausea\"]}"
```

---

## Validation

Run tests:

```powershell
pytest -q tests
```

Run syntax checks:

```powershell
python -m compileall app.py api.py inference.py main.py scripts models tests
```

CI workflow: `.github/workflows/ci.yml`

---

## Evaluation

```powershell
python scripts/evaluation.py
```

Generated outputs are stored under `evaluation_results/`:
- comparative metrics
- per-model reports
- confusion matrices
- SHAP/LIME artifacts (when available)

---

## Constraints

This repository is an educational triage prototype and is **not** a diagnostic medical device.  
Clinical decisions must not be based solely on model output.
Additional context is documented in `MODEL_CARD.md`.
