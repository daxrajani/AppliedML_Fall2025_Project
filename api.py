from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import MIN_SYMPTOMS, load_resources, predict_from_symptoms


class PredictRequest(BaseModel):
    symptoms: List[str] = Field(default_factory=list, description="List of symptom strings.")


class HealthResponse(BaseModel):
    status: str
    min_symptoms_required: int
    model_version: str


class VersionResponse(BaseModel):
    api_version: str
    model_version: str
    generated_at_utc: str


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
    resources = get_resources()
    return HealthResponse(
        status="ok",
        min_symptoms_required=MIN_SYMPTOMS,
        model_version=resources["model_version"],
    )


@app.get("/version", response_model=VersionResponse)
def version_info():
    resources = get_resources()
    return VersionResponse(
        api_version=app.version,
        model_version=resources["model_version"],
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/symptoms")
def list_symptoms():
    resources = get_resources()
    return {"symptoms": resources["feature_names"]}


@app.post("/predict")
def predict(payload: PredictRequest):
    resources = get_resources()
    request_id = str(uuid4())
    result = predict_from_symptoms(
        symptoms=payload.symptoms,
        model=resources["model"],
        feature_names=resources["feature_names"],
        disease_names=resources["disease_names"],
        symptom_synonyms=resources["symptom_synonyms"],
        inconclusive_threshold=resources["confidence_threshold"],
        top2_margin_threshold=resources["top2_margin_threshold"],
        model_version=resources["model_version"],
    )
    result["request_id"] = request_id

    if not result["ok"] and result["status"] == "insufficient_symptoms":
        raise HTTPException(status_code=422, detail=result)

    return result
