# Health Symptom Analyzer

Applied Machine Learning project for disease prediction from symptom inputs.

## Overview

This repository contains an end-to-end ML workflow that:

- trains multiple classification models on `Prototype.csv`
- saves trained models under `saved_models_main/`
- combines tuned models through a soft-voting ensemble
- offers two interfaces:
  - command-line flow (`main.py`)
  - Streamlit web app (`app.py`)

## Project Structure

- `main.py`: model training/loading and CLI prediction flow
- `app.py`: Streamlit app for interactive symptom-based assessment
- `models/`: individual model configuration helpers
- `scripts/evaluation.py`: comparative metrics, confusion matrices, SHAP/LIME outputs
- `Prototype.csv`: training dataset
- `available_symptoms.txt`, `disease_names.txt`: generated lookup files
- `saved_models_main/`: persisted trained models
- `evaluation_results/`: generated evaluation artifacts

## Setup

1. Create and activate a virtual environment.

   Windows (PowerShell):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

### 1) Train or load models + CLI prediction

```powershell
python main.py
```

- If model files already exist in `saved_models_main/`, they are loaded.
- Missing model files are trained and then saved.

### 2) Run web interface

```powershell
streamlit run app.py
```

Then open `http://localhost:8501`.

### 3) Run evaluation script

```powershell
python scripts/evaluation.py
```

This writes evaluation reports into `evaluation_results/`.

## Notes

- Provide at least 3 valid symptoms for more reliable prediction behavior.
- This is an educational ML project and **not** a medical diagnostic system.

## Team

- Dax Rajani
- Harsh Ahuja
- Charanish Miriyala
