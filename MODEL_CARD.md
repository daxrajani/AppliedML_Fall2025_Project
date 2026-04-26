# Model Card: Health Symptom Analyzer (Educational)

## Model Details

- **Task:** Multi-class classification from binary symptom vectors
- **Primary model:** Soft Voting Classifier (KNN, Naive Bayes, Decision Tree, Random Forest, SVM, Logistic Regression)
- **Standalone benchmark:** XGBoost
- **Training data file:** `Prototype.csv`
- **Inference interfaces:** Streamlit UI (`app.py`) and FastAPI service (`api.py`)

## Intended Use

- Educational triage support and machine learning demonstration.
- Portfolio/interview project to show ML workflow, evaluation, and deployment basics.

## Out-of-Scope / Not Intended

- Clinical diagnosis, emergency decision support, or treatment recommendation.
- Any use where medical professionals are not involved in final decisions.

## Data and Label Notes

- Input features are symptom presence indicators.
- Output labels are condition names from the training dataset.
- Label order is controlled via `LabelEncoder` and exported to `disease_names.txt`.

## Performance and Evaluation

- Comparative metrics and confusion matrices are generated with:
  - `python scripts/evaluation.py`
- Outputs are written to `evaluation_results/`.
- Practical interpretation should emphasize macro metrics, confusion pairs, and uncertainty behavior.

## Safety and Risk Considerations

- The app requires a minimum symptom count before prediction.
- Predictions below configured confidence threshold are marked **inconclusive**.
- Top-3 predictions are shown to reduce overconfidence in a single label.
- Explicit disclaimer is included in UI and README.

## Known Limitations

- Dataset quality and representativeness are limited.
- Symptom-only modeling misses demographics, history, vitals, and test results.
- Class imbalance and label noise can impact reliability.
- Confidence scores are not clinical probabilities.

## Maintenance Checklist

- Retrain after any dataset/schema updates.
- Keep feature ordering stable (`feature_names_in_` mapping is used for inference).
- Run CI tests and evaluation script before releasing updates.
