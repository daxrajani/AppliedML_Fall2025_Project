from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

MODEL_PATH = Path("saved_models_main") / "voting_clf.pkl"
SYMPTOMS_FILE = Path("available_symptoms.txt")
DISEASES_FILE = Path("disease_names.txt")

MIN_SYMPTOMS = 3
INCONCLUSIVE_THRESHOLD = 0.30


def _read_lines(file_path: Path) -> List[str]:
    if not file_path.exists():
        return []
    return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_resources() -> Dict[str, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'. Run 'python main.py' first.")

    model = joblib.load(MODEL_PATH)
    disease_names = _read_lines(DISEASES_FILE)
    symptom_file_list = _read_lines(SYMPTOMS_FILE)

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = symptom_file_list

    if not feature_names:
        raise ValueError("No feature names available from model or symptom file.")
    if not disease_names:
        raise ValueError("No diseases found. Ensure 'disease_names.txt' exists and is non-empty.")

    synonyms = {
        "fever": "high_fever",
        "temperature": "high_fever",
        "tiredness": "fatigue",
        "stomachache": "stomach_pain",
    }

    return {
        "model": model,
        "feature_names": feature_names,
        "disease_names": disease_names,
        "symptom_file_list": symptom_file_list,
        "symptom_synonyms": synonyms,
    }


def normalize_symptom(symptom: str, synonyms: Dict[str, str] | None = None) -> str:
    cleaned = symptom.strip().lower().replace(" ", "_").replace("'", "")
    if synonyms:
        cleaned = synonyms.get(cleaned, cleaned)
    return cleaned


def predict_from_symptoms(
    symptoms: List[str],
    model,
    feature_names: List[str],
    disease_names: List[str],
    symptom_synonyms: Dict[str, str] | None = None,
    min_symptoms: int = MIN_SYMPTOMS,
    inconclusive_threshold: float = INCONCLUSIVE_THRESHOLD,
) -> Dict[str, object]:
    normalized = [normalize_symptom(symptom, symptom_synonyms) for symptom in symptoms if symptom and symptom.strip()]
    normalized = [symptom for symptom in normalized if symptom]

    valid_set = set(feature_names)
    unique_valid_symptoms: List[str] = []
    ignored_symptoms: List[str] = []
    seen = set()

    for symptom in normalized:
        if symptom in valid_set:
            if symptom not in seen:
                unique_valid_symptoms.append(symptom)
                seen.add(symptom)
        else:
            ignored_symptoms.append(symptom)

    if len(unique_valid_symptoms) < min_symptoms:
        return {
            "ok": False,
            "status": "insufficient_symptoms",
            "message": f"Please provide at least {min_symptoms} valid distinct symptoms.",
            "selected_symptoms": unique_valid_symptoms,
            "ignored_symptoms": ignored_symptoms,
            "top_predictions": [],
            "predicted_disease": None,
            "confidence": 0.0,
            "is_inconclusive": True,
        }

    input_vector = [1 if feature in unique_valid_symptoms else 0 for feature in feature_names]
    input_df = pd.DataFrame([input_vector], columns=feature_names)
    probabilities = list(model.predict_proba(input_df)[0])
    top_indices = sorted(range(len(probabilities)), key=lambda idx: probabilities[idx], reverse=True)[:3]
    top_predictions = [
        {
            "disease": disease_names[int(index)],
            "confidence": float(probabilities[int(index)]),
        }
        for index in top_indices
    ]

    best = top_predictions[0]
    is_inconclusive = best["confidence"] < inconclusive_threshold

    return {
        "ok": True,
        "status": "inconclusive" if is_inconclusive else "ok",
        "message": "Prediction generated successfully.",
        "selected_symptoms": unique_valid_symptoms,
        "ignored_symptoms": ignored_symptoms,
        "top_predictions": top_predictions,
        "predicted_disease": best["disease"],
        "confidence": best["confidence"],
        "is_inconclusive": is_inconclusive,
    }
