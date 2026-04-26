from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import MIN_SYMPTOMS, load_resources, predict_from_symptoms


class PredictRequest(BaseModel):
    symptoms: List[str] = Field(default_factory=list, description="List of symptom strings.")


class HealthResponse(BaseModel):
    status: str
    min_symptoms_required: int


app = FastAPI(
    title="Health Symptom Analyzer API",
    description="Educational symptom triage API built on a voting ensemble model.",
    version="1.0.0",
)

_resources = None


def get_resources():
    global _resources
    if _resources is None:
        _resources = load_resources()
    return _resources


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", min_symptoms_required=MIN_SYMPTOMS)


@app.get("/symptoms")
def list_symptoms():
    resources = get_resources()
    return {"symptoms": resources["feature_names"]}


@app.post("/predict")
def predict(payload: PredictRequest):
    resources = get_resources()
    result = predict_from_symptoms(
        symptoms=payload.symptoms,
        model=resources["model"],
        feature_names=resources["feature_names"],
        disease_names=resources["disease_names"],
        symptom_synonyms=resources["symptom_synonyms"],
    )

    if not result["ok"] and result["status"] == "insufficient_symptoms":
        raise HTTPException(status_code=422, detail=result)

    return result
