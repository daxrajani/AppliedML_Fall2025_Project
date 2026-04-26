# Health Symptom Analyzer

An end-to-end machine learning project that predicts likely conditions from symptom inputs using an ensemble classifier, with both web UI and API interfaces.

## Why This Project Stands Out

- Multi-model training pipeline with a weighted voting ensemble
- Two deployment surfaces: Streamlit app and FastAPI service
- Confidence-aware inference with **top-3 outputs** and **inconclusive handling**
- Evaluation workflow with metrics, confusion matrices, SHAP, and LIME
- Basic production engineering: shared inference module, tests, and CI

## Architecture

- `main.py`: trains/loads models and supports CLI predictions
- `inference.py`: shared inference logic (normalization, vectorization, thresholds)
- `app.py`: Streamlit interface
- `api.py`: FastAPI endpoints (`/health`, `/symptoms`, `/predict`)
- `scripts/evaluation.py`: quantitative + qualitative evaluation reports
- `models/`: individual model builder functions

## Quickstart

1. Create and activate environment (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train/load models:

```powershell
python main.py
```

3. Run Streamlit app:

```powershell
streamlit run app.py
```

4. Run API:

```powershell
uvicorn api:app --reload
```

## API Example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"symptoms\": [\"high_fever\", \"chills\", \"headache\", \"nausea\"]}"
```

## Testing and Quality

- Run unit tests:

```powershell
pytest -q
```

- GitHub Actions workflow at `.github/workflows/ci.yml` runs tests + syntax checks on push/PR.

## Evaluation

Generate model performance artifacts:

```powershell
python scripts/evaluation.py
```

Outputs are saved in `evaluation_results/`.

## Safety and Disclaimer

This project is an educational ML triage prototype. It is **not** a medical diagnosis system and must not replace clinical judgment. See `MODEL_CARD.md` for intended use, risks, and limitations.

## Team

- Dax Rajani
- Harsh Ahuja
- Charanish Miriyala
