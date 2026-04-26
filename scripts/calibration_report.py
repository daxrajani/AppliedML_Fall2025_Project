import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def expected_calibration_error(y_true_binary, probs, bins=10):
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if np.any(mask):
            acc = y_true_binary[mask].mean()
            conf = probs[mask].mean()
            ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


def main():
    root = Path(".")
    output_dir = root / "evaluation_results" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = root / "saved_models_main" / "voting_clf.pkl"
    if not model_path.exists():
        raise FileNotFoundError("saved_models_main/voting_clf.pkl not found. Run training first.")

    df = pd.read_csv(root / "Prototype.csv")
    feature_names = [
        line.strip()
        for line in (root / "available_symptoms.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected features in dataset: {missing_features[:5]}")

    X = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = (X > 0).astype(int)

    le = LabelEncoder()
    y = le.fit_transform(df["prognosis"].astype(str))

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    model = joblib.load(model_path)
    probas = model.predict_proba(X_test)
    preds = np.argmax(probas, axis=1)
    top_probs = probas[np.arange(len(preds)), preds]
    correctness = (preds == y_test).astype(int)

    frac_pos, mean_pred = calibration_curve(correctness, top_probs, n_bins=10, strategy="uniform")
    ece = expected_calibration_error(correctness, top_probs, bins=10)
    brier = brier_score_loss(correctness, top_probs)
    ll = log_loss(y_test, probas)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "--", label="Ideal")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title("Reliability Curve (Top-class confidence)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reliability_curve.png")
    plt.close()

    report_lines = [
        "Calibration Report",
        "==================",
        f"ECE (10 bins): {ece:.4f}",
        f"Brier score: {brier:.4f}",
        f"Multiclass log-loss: {ll:.4f}",
        f"Sample count: {len(X_test)}",
    ]
    (output_dir / "calibration_metrics.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
