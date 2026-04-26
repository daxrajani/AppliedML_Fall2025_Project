# Health Symptom Analyzer

Symptom-based multi-class classification system with:
- calibrated ensemble training
- shared inference logic
- Streamlit UI
- FastAPI service
- automated test and CI validation

## Architecture

- `main.py`
  - data loading and cleaning
  - train/validation/test splitting
  - base model training
  - calibrated ensemble training
  - escalation threshold tuning
  - model manifest generation
- `inference.py`
  - runtime normalization and validation
  - symptom-to-feature mapping
  - top-k ranking
  - escalation policy (confidence + top-2 margin)
- `app.py`
  - interactive prediction UI
  - confidence and escalation display
- `api.py`
  - REST endpoints for health, version, symptom list, prediction
- `models/`
  - model factory functions and tuned estimator settings
- `scripts/`
  - `evaluation.py` (classification metrics and interpretability artifacts)
  - `calibration_report.py` (reliability curve, ECE, Brier score, log-loss)
- `data_generation/`
  - `augment_prototype.py` (optional synthetic expansion dataset generation)

## Data Inputs and Outputs

### Input datasets
- `Prototype.csv` (base dataset)
- `Prototype_augmented.csv` (optional, generated)

### Generated metadata
- `available_symptoms.txt`
- `disease_names.txt`
- `saved_models_main/model_manifest.json`

### Model artifacts
- `saved_models_main/*.pkl`

## Training Pipeline

1. Load base dataset (and optional augmented dataset when enabled).
2. Normalize labels and coerce symptom features to binary values.
3. Drop constant symptom columns.
4. Split data into train/validation/test.
5. Train base estimators (`KNN`, `Naive Bayes`, `Decision Tree`, `Random Forest`, `SVM`, `Logistic Regression`, `XGBoost`).
6. Train calibrated soft-voting ensemble.
7. Tune escalation thresholds on validation set:
   - absolute confidence threshold
   - top-2 margin threshold
8. Evaluate on test set and write model manifest.

## Inference Contract

### Input
- symptom string list

### Output fields (core)
- `predicted_disease`
- `top_predictions` (ranked with confidence)
- `confidence`
- `top2_margin`
- `is_inconclusive`
- `requires_clinician_review`
- `escalation_reason`
- `model_version`

## Local Setup (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

### Train on base data
```powershell
python main.py
```

### Optional: generate and use augmented data
```powershell
python data_generation/augment_prototype.py
$env:USE_AUGMENTED_DATA="1"
python main.py
```

### Run Streamlit UI
```powershell
streamlit run app.py
```

### Run API
```powershell
uvicorn api:app --reload
```

## API Endpoints

- `GET /health`
- `GET /version`
- `GET /symptoms`
- `POST /predict`

### Example
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"symptoms\": [\"high_fever\", \"chills\", \"headache\", \"nausea\"]}"
```

## Evaluation

```powershell
python scripts/evaluation.py
python scripts/calibration_report.py
```

Artifacts are written under `evaluation_results/`.

## Validation

```powershell
pytest -q tests
python -m compileall app.py api.py inference.py main.py scripts models tests data_generation
```

CI configuration: `.github/workflows/ci.yml`

## Operational Notes

- No Docker is required.
- Large model binaries may exceed GitHub size limits; retrain locally using `python main.py` if needed.
- This is a triage-oriented educational system, not a clinical diagnostic device.
