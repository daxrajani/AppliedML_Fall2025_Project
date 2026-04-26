from inference import predict_from_symptoms


class DummyModel:
    def __init__(self, probabilities):
        self._probabilities = probabilities

    def predict_proba(self, _df):
        return [self._probabilities]


def test_predict_inconclusive_with_low_confidence():
    model = DummyModel([0.2, 0.15, 0.65])
    result = predict_from_symptoms(
        symptoms=["itching", "skin_rash", "nodal_skin_eruptions"],
        model=model,
        feature_names=["itching", "skin_rash", "nodal_skin_eruptions"],
        disease_names=["A", "B", "C"],
        inconclusive_threshold=0.70,
    )

    assert result["ok"] is True
    assert result["is_inconclusive"] is True
    assert result["requires_clinician_review"] is True
    assert result["predicted_disease"] == "C"


def test_predict_rejects_insufficient_symptoms():
    model = DummyModel([0.9, 0.1, 0.0])
    result = predict_from_symptoms(
        symptoms=["itching", "skin_rash"],
        model=model,
        feature_names=["itching", "skin_rash", "nodal_skin_eruptions"],
        disease_names=["A", "B", "C"],
    )

    assert result["ok"] is False
    assert result["status"] == "insufficient_symptoms"


def test_predict_handles_duplicates_and_ignored():
    model = DummyModel([0.8, 0.1, 0.1])
    result = predict_from_symptoms(
        symptoms=["itching", "itching", "skin_rash", "unknown_symptom", "nodal_skin_eruptions"],
        model=model,
        feature_names=["itching", "skin_rash", "nodal_skin_eruptions"],
        disease_names=["A", "B", "C"],
    )

    assert result["ok"] is True
    assert result["selected_symptoms"] == ["itching", "skin_rash", "nodal_skin_eruptions"]
    assert "unknown_symptom" in result["ignored_symptoms"]


def test_predict_escalates_on_small_top2_margin():
    model = DummyModel([0.51, 0.49, 0.0])
    result = predict_from_symptoms(
        symptoms=["itching", "skin_rash", "nodal_skin_eruptions"],
        model=model,
        feature_names=["itching", "skin_rash", "nodal_skin_eruptions"],
        disease_names=["A", "B", "C"],
        inconclusive_threshold=0.40,
        top2_margin_threshold=0.05,
    )

    assert result["ok"] is True
    assert result["requires_clinician_review"] is True
    assert result["escalation_reason"] == "low_top2_margin"
